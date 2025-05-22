import os
import json
import numpy as np
from pydantic import BaseModel, Field
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from pydantic_autocli import AutoCLI, param
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Accuracy
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint, LearningRateMonitor
from scipy import ndimage

from .utils import fix_global_seed, CustomEarlyStopping
from .datasets import ConjDataset, COLOR_MAP
from .models import create_segmentation_model
from .losses import get_loss, CrossEntropy


def save_results(results, config, save_dir):
    results_dict = {
        'test_results': results[0],
        'config': config.model_dump()
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Test results saved to: {os.path.join(save_dir, 'results.json')}")


class ConjConfig(BaseModel):
    lr: float = 0.0001
    batch_size: int = param(5, s='-B')
    fold: int = 1
    loss: str = param('ce', choices=['dice', 'focal', 'iou', 'combined'])
    plateau: bool = False
    nopretrained: bool = False
    max_epochs: int = param(100, s='-e', l='--max-epochs')

    model_name: str = param('ternaus16n', l='--model', s='-M')
    size: int = 512


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

class ConjModule(pl.LightningModule):

    def __init__(self, config:ConjConfig):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config

        self.num_classes = 3
        self.unet = create_segmentation_model(
                config.model_name,
                num_classes=self.num_classes,
                pretrained=not config.nopretrained)
        self.criterion = get_loss('ce')

        self.metric_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.metric_jac = JaccardIndex(task='multiclass', num_classes=self.num_classes)


    def forward(self, x):
        # Handle if x is a list or tuple (batch structure)
        if isinstance(x, (list, tuple)):
            x = x[0]  # Take the first element (assuming it's the input tensor)
        h = self.unet(x)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        acc = self.metric_acc(y, t)
        jac = self.metric_jac(y, t)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_jac', jac, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        acc = self.metric_acc(y, t)
        jac = self.metric_jac(y, t)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_jac', jac, on_step=False, on_epoch=True, prog_bar=False)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        acc = self.metric_acc(y, t)
        jac = self.metric_jac(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_jac', jac, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'test_acc': acc, 'test_jac': jac}

    def predict_step(self, batch, batch_idx):
        # Handle different batch structures
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer


# Prediction callback for visualizing and saving outputs
class PredictionWriter(pl.Callback):
    def __init__(self, output_dir, save, num_vis_samples):
        super().__init__()
        self.output_dir = output_dir
        self.save = save
        self.num_vis_samples = num_vis_samples
        self.batch_count = 0

        # Create necessary directories for saving predictions
        if save:
            self.masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(self.masks_dir, exist_ok=True)
            self.logits_dir = os.path.join(output_dir, 'logits')
            os.makedirs(self.logits_dir, exist_ok=True)

            # Store all logits for stacking later
            self.all_logits = []
            # Store file mapping for original filenames
            self.file_indices = []
            # Store the dataset for later use
            self.dataset = None

    def on_predict_start(self, trainer, pl_module):
        if self.save:
            # Get the dataset from the dataloader at the start
            self.dataset = trainer.predict_dataloaders.dataset

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        preds = torch.argmax(outputs, dim=1)

        # Save masks/logits if requested
        if self.save:
            # Get the dataset if not already stored
            if self.dataset is None:
                # Access dataset directly from the dataloader
                self.dataset = trainer.predict_dataloaders.dataset

            # Get indices for this batch
            start_idx = self.batch_count * len(inputs)
            indices = list(range(start_idx, start_idx + len(inputs)))

            # Save masks with original filenames
            for i, pred in enumerate(preds):
                idx = indices[i]
                if idx < len(self.dataset.df):
                    # Get original filename from dataset
                    row = self.dataset.df.iloc[idx]
                    id_str = str(row['test_ID']).zfill(4)
                    rl = row['R/L']
                    original_filename = f'{id_str}_{rl}_01.png'

                    # Convert prediction to colored mask
                    colored_mask = COLOR_MAP[pred.cpu().numpy()]

                    # Create PIL image and save with original filename
                    img = Image.fromarray(colored_mask.astype(np.uint8))
                    img.save(os.path.join(self.masks_dir, original_filename))

                    # Store index mapping for logits
                    self.file_indices.append((idx, original_filename))

            # Store logits for later stacking
            self.all_logits.append(outputs.cpu())

        self.batch_count += 1

    def on_predict_end(self, trainer, pl_module):
        if self.save and self.all_logits:
            # Stack all logits
            stacked_logits = torch.cat(self.all_logits, dim=0)

            # Save as a single file
            np.save(os.path.join(self.logits_dir, 'all_logits.npy'), stacked_logits.numpy())

            # Save index mapping for reference
            mapping = {idx: filename for idx, filename in self.file_indices}
            with open(os.path.join(self.logits_dir, 'file_mapping.json'), 'w') as f:
                json.dump(mapping, f, indent=2)


class CLI(AutoCLI):

    class CommonArgs(BaseModel):
        seed: int = 0
        device: str = param('cuda', help="Device to use for inference: 'cpu' or 'cuda'")

    def pre_common(self, a:CommonArgs):
        fix_global_seed(a.seed)
        pl.seed_everything(a.seed)
        torch.set_float32_matmul_precision('medium')
        # matplotlib.use('QtAgg')

    class ModelArgs(CommonArgs):
        model_name: str = param('unet16', s='-M', l='--model')

    def run_model(self, a):
        m = create_model(a.model_name, num_classes=3)
        t = torch.randn(2, 3, 256, 256)
        print(m(t).shape)

    def run_image(self, a):
        ds = ConjDataset(fold=1, mode='val', augmentation=True, normalization=False)
        print(ds, len(ds))
        count = 5
        for i in range(count):
            image = ds.images[i]
            x, y = ds[i]
            print('x', type(x), x.shape)
            print('y', type(y), y.shape)
            plt.subplot(count,3,3*i+1)
            plt.imshow(image)
            plt.subplot(count,3,3*i+2)
            # plt.imshow(Image.fromarray((x.numpy().transpose(1, 2, 0)*255).astype(np.uint8)))
            plt.imshow(x.numpy().transpose(1, 2, 0))
            plt.subplot(count,3,3*i+3)
            plt.imshow(y.numpy())
        plt.show()

    class TrainArgs(CommonArgs, ConjConfig):
        num_workers: int = 4
        checkpoint_dir: str = 'checkpoints'

    def run_train(self, a:TrainArgs):
        config = ConjConfig(**a.model_dump())

        # Create experiment name and save directory
        exp_name = f'{a.model_name}_fold{a.fold}'
        save_dir = os.path.join(a.checkpoint_dir, 'conj', exp_name)
        os.makedirs(save_dir, exist_ok=True)

        train_ds = ConjDataset(fold=a.fold, mode='train', augmentation=True)
        val_ds = ConjDataset(fold=a.fold, mode='val', augmentation=False)
        train_loader = DataLoader(train_ds, a.batch_size, num_workers=a.num_workers, shuffle=True)
        val_loader = DataLoader(val_ds, a.batch_size, num_workers=a.num_workers)

        # Create logger first to get version
        logger = TensorBoardLogger(os.path.join(a.checkpoint_dir, 'conj'), name=exp_name, sub_dir='logs')
        version_dir = os.path.join(save_dir, f'version_{logger.version}')
        os.makedirs(version_dir, exist_ok=True)

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath=version_dir,
            filename='{epoch:02d}-{val_loss:.3f}',
            save_top_k=1,
            mode='min',
            save_weights_only=True
        )

        early_stopping = CustomEarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        accelerator = 'gpu' if a.device == 'cuda' else 'cpu'

        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            devices=1,
            accelerator=accelerator,
            benchmark=False,
            callbacks=[CustomProgressBar(), checkpoint, early_stopping, lr_monitor],
            log_every_n_steps=1,
            logger=logger,
        )

        print('config', config)
        module = ConjModule(config)
        trainer.fit(module, train_loader, val_loader)

        print(f'\nBest model path: {checkpoint.best_model_path}')

        # Restore best model
        module = ConjModule.load_from_checkpoint(checkpoint.best_model_path)

        test_ds = ConjDataset(fold=module.config.fold, mode='val', size=640, augmentation=False)
        test_loader = DataLoader(test_ds, a.batch_size, num_workers=a.num_workers)
        results = trainer.test(module, test_loader)

        # Save results to JSON
        save_results(results, config, save_dir)


    class PredictArgs(CommonArgs):
        checkpoint: str = param(..., s='-C')
        batch_size: int = param(16, s='-B')
        num_workers: int = 4
        output_dir: str = ''
        save: bool = False
        num_vis_samples: int = param(5, l='--num', s='-n')

    def run_predict(self, a):
        # Load model from checkpoint
        module = ConjModule.load_from_checkpoint(a.checkpoint, map_location=torch.device(a.device))
        print(module.config)

        # Determine experiment directory
        checkpoint_dir = os.path.dirname(a.checkpoint)

        # Create output directory - default to {checkpoint_dir}/results
        output_dir = a.output_dir if a.output_dir else os.path.join(checkpoint_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)

        # Load test dataset
        test_ds = ConjDataset(fold=module.config.fold, mode='val', size=640, augmentation=False)
        test_loader = DataLoader(test_ds, a.batch_size, num_workers=a.num_workers)

        # Use CPU or GPU based on device argument
        accelerator = 'gpu' if a.device == 'cuda' else 'cpu'

        # Create a trainer for prediction only
        predict_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            callbacks=[PredictionWriter(
                output_dir=output_dir,
                save=a.save,
                num_vis_samples=a.num_vis_samples
            )]
        )

        # 予測実行 (テストではなく予測モードで実行)
        predictions = predict_trainer.predict(module, test_loader)

        # テストメトリクスも取得
        results = predict_trainer.test(module, test_loader)

        # Save results to JSON
        save_results(results, module.config, output_dir)

        print(f"Prediction completed. Results saved to {output_dir}")

    class EvaluateArgs(CommonArgs):
        prediction_dir: str = param(..., s='-p', help="Directory containing prediction results")
        largest: bool = param(False, help="Use largest blob for evaluation")
        output_file: str = param('evaluation_results.json', help="Output file for evaluation results")

    def run_evaluate(self, a):
        # Get paths to prediction files
        prediction_dir = a.prediction_dir
        logits_file = os.path.join(prediction_dir, 'logits', 'all_logits.npy')
        mapping_file = os.path.join(prediction_dir, 'logits', 'file_mapping.json')

        if not os.path.exists(logits_file) or not os.path.exists(mapping_file):
            print(f"Error: Could not find prediction files in {prediction_dir}")
            return False

        # Load logits and mapping
        print(f"Loading logits from {logits_file}")
        logits = np.load(logits_file)

        with open(mapping_file, 'r') as f:
            file_mapping = json.load(f)

        # Load dataset to get ground truth
        # Extract fold from the prediction directory path
        parts = prediction_dir.split('fold')
        if len(parts) > 1:
            fold = int(parts[1][0])  # Assuming fold is a single digit after "fold"
        else:
            fold = 1  # Default to fold 1

        print(f"Using fold {fold} for evaluation")
        test_ds = ConjDataset(fold=fold, mode='val', size=640, augmentation=False)

        # Convert logits to predictions
        preds = np.argmax(logits, axis=1)

        # Get ground truth masks
        gt_masks = [mask for mask in test_ds.masks]

        # Calculate metrics
        results = {}

        # Standard metrics (without largest blob)
        acc, jac = self.calculate_metrics(preds, gt_masks)
        results['standard'] = {
            'accuracy': acc,
            'jaccard': jac
        }

        # If requested, also calculate metrics with largest blob
        if a.largest:
            # Apply largest blob processing
            processed_preds = self.apply_largest_blob(preds)
            acc_largest, jac_largest = self.calculate_metrics(processed_preds, gt_masks)
            results['largest_blob'] = {
                'accuracy': acc_largest,
                'jaccard': jac_largest
            }

        # Save results
        output_path = os.path.join(prediction_dir, a.output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print results
        print("\nEvaluation Results:")
        print("-------------------")
        print(f"Standard Metrics:")
        print(f"  Accuracy: {results['standard']['accuracy']:.4f}")
        print(f"  Jaccard:  {results['standard']['jaccard']:.4f}")

        if a.largest:
            print(f"\nLargest Blob Metrics:")
            print(f"  Accuracy: {results['largest_blob']['accuracy']:.4f}")
            print(f"  Jaccard:  {results['largest_blob']['jaccard']:.4f}")

        print(f"\nResults saved to {output_path}")
        return results

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate accuracy and Jaccard index for predictions"""
        total_acc = 0
        total_jac = 0
        count = 0

        jaccard = JaccardIndex(task='multiclass', num_classes=3)

        for i, gt in enumerate(ground_truth):
            if i >= len(predictions):
                break

            pred = predictions[i]

            # Convert to tensors if needed
            if not isinstance(pred, torch.Tensor):
                pred = torch.tensor(pred)
            if not isinstance(gt, torch.Tensor):
                gt = torch.tensor(gt)

            # Calculate metrics
            acc = (pred == gt).float().mean().item()
            jac = jaccard(pred, gt).item()

            total_acc += acc
            total_jac += jac
            count += 1

        return total_acc / count if count > 0 else 0, total_jac / count if count > 0 else 0

    def apply_largest_blob(self, predictions):
        """Apply largest blob processing to predictions"""
        processed = []

        for pred in predictions:
            # Process each class separately
            result = np.zeros_like(pred)

            for class_id in range(1, 3):  # Process class 1 and 2 (skip background)
                # Create binary mask for this class
                binary = (pred == class_id).astype(np.uint8)

                # Find connected components
                labeled, num_features = ndimage.label(binary)

                if num_features > 0:
                    # Find largest component
                    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
                    largest_label = np.argmax(sizes) + 1

                    # Keep only the largest component
                    largest_mask = (labeled == largest_label)

                    # Add to result
                    result[largest_mask] = class_id

            processed.append(result)

        return processed


if __name__ == '__main__':
    cli = CLI()
    cli.run()
