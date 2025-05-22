import os
import numpy as np
from pydantic import BaseModel, Field
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
import json


from .utils import fix_global_seed
from .datasets import ConjDataset
from .models import create_model
from .losses import get_loss, CrossEntropy


class ConjConfig(BaseModel):
    lr: float = 0.0001
    batch_size: int = param(5, s='-B')
    fold: int = 1
    loss: str = param('ce', choices=['dice', 'focal', 'iou', 'combined'])
    plateau: bool = False
    nopretrained: bool = False
    max_epochs: int = param(100, s='-e', l='--max-epochs')

    arch_name: str = param('ternaus16n', l='--arch', s='-A')
    size: int = 512


class CustomEarlyStopping(EarlyStopping):
    def _improvement_message(self, *args, **kwargs):
        return '\n' + super()._improvement_message(*args, **kwargs)

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
        self.unet = create_model(config.arch_name,
                                 num_classes=self.num_classes,
                                 pretrained=not config.nopretrained)
        self.criterion = get_loss('ce')

        self.metric_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.metric_jac = JaccardIndex(task='multiclass', num_classes=self.num_classes)


    def forward(self, x):
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


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    # def on_before_optimizer_step(self, optimizer):
    #     opt = optimizer
    #     for param_group in opt.param_groups:
    #         current_lr = param_group['lr']
    #         self.log('lr', current_lr, prog_bar=True)

class CLI(AutoCLI):

    class CommonArgs(BaseModel):
        seed: int = 0

    def pre_common(self, a:CommonArgs):
        fix_global_seed(a.seed)
        pl.seed_everything(a.seed)
        torch.set_float32_matmul_precision('medium')
        # matplotlib.use('QtAgg')

    class ModelArgs(CommonArgs):
        arch_name: str = param('unet16', s='-A', l='--arch')

    def run_model(self, a):
        m = create_model(a.arch_name, num_classes=3)
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
        exp_name = f'{a.arch_name}_fold{a.fold}'
        save_dir = os.path.join(a.checkpoint_dir, 'conj', exp_name)
        os.makedirs(save_dir, exist_ok=True)

        train_ds = ConjDataset(fold=a.fold, mode='train', augmentation=True)
        val_ds = ConjDataset(fold=a.fold, mode='val', augmentation=False)
        train_loader = DataLoader(train_ds, a.batch_size, num_workers=a.num_workers, shuffle=True)
        val_loader = DataLoader(val_ds, a.batch_size, num_workers=a.num_workers)

        # Create logger first to get version
        logger = TensorBoardLogger(os.path.join(a.checkpoint_dir, 'conj'), name=exp_name)
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

        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            devices=1,
            accelerator='gpu',
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

        test_ds = ConjDataset(mode='val', size=640, augmentation=False)
        test_loader = DataLoader(test_ds, a.batch_size, num_workers=a.num_workers)
        results = trainer.test(module, test_loader)
        
        # Save results to JSON
        results_dict = {
            'test_results': results[0],
            'config': config.model_dump()
        }
        with open(os.path.join(version_dir, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)


    class PredictArgs(CommonArgs):
        checkpoint: str = param(..., s='-c')
        batch_size: int = param(16, s='-B')
        num_workers: int = 4
        device: str = 'cuda'

    def run_predict(self, a):
        module = ConjModule.load_from_checkpoint(
            a.checkpoint,
        )
        print(module.config)

        test_ds = ConjDataset(mode='val', size=640, augmentation=False)
        test_loader = DataLoader(test_ds, a.batch_size, num_workers=a.num_workers)

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
        )
        results = trainer.test(module, test_loader)
        print(results)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
