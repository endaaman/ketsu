import os
import numpy as np
from pydantic import BaseModel, Field
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import timm
from pydantic_autocli import AutoCLI, param
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Accuracy
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint, LearningRateMonitor
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .utils import fix_global_seed
from .datasets import SpotsDataset


def to_coral_labels(y, num_classes=4):
    """
    整数ラベル (0, 1, 2, 3) をCORALラベル形式に変換

    Parameters:
    -----------
    y : torch.Tensor or numpy.ndarray
        整数ラベル (0, 1, 2, 3)
    num_classes : int
        クラス数

    Returns:
    --------
    torch.Tensor
        CORAL形式のラベル
    """
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    # 閾値のテンソルを作成 (num_classes-1,)
    thresholds = torch.arange(num_classes-1, device=y.device)

    # ブロードキャストで比較 (..., num_classes-1)
    return (y.unsqueeze(-1) > thresholds).float()


class CoralModel(nn.Module):
    def __init__(self, base, num_classes=4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.base = timm.create_model(base, pretrained=pretrained, num_classes=1)
        self.coral_biases = nn.Parameter(torch.arange(0, num_classes-1, dtype=torch.float32).view(1, -1))

    def forward(self, x):
        logits = self.base(x)
        return logits + self.coral_biases


class CoralConfig(BaseModel):
    fold: int = param(1, s='-f', l='--fold')
    model_name: str = param('resnet34', s='-M', l='--model')
    num_classes: int = 4
    pretrained: bool = True
    lr: float = param(1e-4, s='-l', l='--lr')
    batch_size: int = param(32, s='-B', l='--batch-size')
    max_epochs: int = param(100, s='-e', l='--max-epochs')
    early_stopping_patience: int = 10


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class CoralLightningModule(pl.LightningModule):
    def __init__(self, config: CoralConfig):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config
        self.model = CoralModel(config.model_name, num_classes=config.num_classes, pretrained=config.pretrained)
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric_acc = Accuracy(task='multiclass', num_classes=config.num_classes)

    def forward(self, x):
        return self.model(x)

    def _get_prediction(self, logits):
        # CORALの予測をクラスに変換
        # logits: (B, num_classes-1) -> pred: (B,)
        return (logits > 0).sum(dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        logits = self(x)
        loss = self.criterion(logits, y_coral)
        
        # 予測とaccuracyの計算
        pred = self._get_prediction(logits)
        acc = self.metric_acc(pred, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        logits = self(x)
        loss = self.criterion(logits, y_coral)
        
        # 予測とaccuracyの計算
        pred = self._get_prediction(logits)
        acc = self.metric_acc(pred, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        logits = self(x)
        loss = self.criterion(logits, y_coral)
        
        # 予測とaccuracyの計算
        pred = self._get_prediction(logits)
        acc = self.metric_acc(pred, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)


class CLI(AutoCLI):
    class CommonArgs(BaseModel):
        seed: int = 0

    def pre_common(self, a: CommonArgs):
        fix_global_seed(a.seed)
        pl.seed_everything(a.seed)
        torch.set_float32_matmul_precision('medium')

    class TrainArgs(CommonArgs, CoralConfig):
        num_workers: int = param(4, s='-w', l='--num-workers')

    def run_train(self, a: TrainArgs):
        config = CoralConfig(**a.model_dump())

        # Create experiment name and save directory
        exp_name = f'coral_{a.model_name}_fold{a.fold}'
        save_dir = os.path.join('checkpoints', 'spots', exp_name)
        os.makedirs(save_dir, exist_ok=True)

        # Create datasets and dataloaders
        train_dataset = SpotsDataset(fold=a.fold, mode='train', augmentation=True)
        val_dataset = SpotsDataset(fold=a.fold, mode='val', augmentation=False)
        train_loader = DataLoader(train_dataset, a.batch_size, num_workers=a.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, a.batch_size, num_workers=a.num_workers)

        # Create logger first to get version
        logger = TensorBoardLogger(os.path.join('checkpoints', 'spots'), name=exp_name)
        version_dir = os.path.join(save_dir, f'version_{logger.version}')
        os.makedirs(version_dir, exist_ok=True)

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, mode='min'),
            ModelCheckpoint(
                dirpath=version_dir,  # TensorBoardのversion_%dフォルダに保存
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='{epoch}-{val_loss:.4f}'
            ),
            LearningRateMonitor(logging_interval='epoch'),
            CustomProgressBar()
        ]

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices=1
        )

        # Train
        module = CoralLightningModule(config)
        trainer.fit(module, train_loader, val_loader)

        # 学習後の評価と結果の保存
        results = trainer.test(module, val_loader)
        
        # 結果をJSONとして保存
        results_dict = {
            'test_results': results[0],
            'config': config.model_dump()
        }
        with open(os.path.join(version_dir, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)

    class ModelArgs(CommonArgs):
        model_name: str = param('resnet34', s='-M', l='--model')

    def run_model(self, a: ModelArgs):
        model = CoralModel(a.model_name, num_classes=4, pretrained=False)
        t = torch.randn(2, 3, 256, 256)
        print(model(t).shape)

    class TestArgs(CommonArgs):
        checkpoint: str = param(..., s='-c')
        batch_size: int = param(32, s='-b')
        num_workers: int = 4

    def run_test(self, a: TestArgs):
        module = CoralLightningModule.load_from_checkpoint(a.checkpoint)
        test_dataset = SpotsDataset(fold=module.config.fold, mode='val', augmentation=False)
        test_loader = DataLoader(test_dataset, a.batch_size, num_workers=a.num_workers)

        trainer = pl.Trainer(accelerator='auto', devices=1)
        results = trainer.test(module, test_loader)
        print(results)

    class PredictArgs(CommonArgs):
        checkpoint: str = param(..., s='-c')
        batch_size: int = param(32, s='-b')
        num_workers: int = 4

    def run_predict(self, a: PredictArgs):
        module = CoralLightningModule.load_from_checkpoint(a.checkpoint)
        print('Config:', module.config)

        # デバイスの設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        module = module.to(device)
        module.eval()

        # チェックポイントのパスから実験ディレクトリを取得
        exp_dir = os.path.dirname(a.checkpoint)
        version_dir = os.path.dirname(exp_dir)  # version_0 の親ディレクトリ
        version = os.path.basename(exp_dir).split('_')[1]  # version_0 から 0 を取得

        test_dataset = SpotsDataset(fold=module.config.fold, mode='val', augmentation=False)
        test_loader = DataLoader(test_dataset, a.batch_size, num_workers=a.num_workers)
        
        # 予測を収集
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                logits = module(x)
                preds = module._get_prediction(logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 結果を表示
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 混同行列
        cm = confusion_matrix(all_labels, all_preds)
        print('\nConfusion Matrix:')
        print(cm)
        
        # 混同行列のプロット
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(exp_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 各クラスの予測分布
        print('\nPrediction Distribution:')
        class_dist = {}
        for i in range(module.config.num_classes):
            pred_count = (all_preds == i).sum()
            true_count = (all_labels == i).sum()
            print(f'Class {i}: Predicted {pred_count}, Actual {true_count}')
            class_dist[f'class_{i}'] = {
                'predicted': int(pred_count),
                'actual': int(true_count)
            }
        
        # 予測分布のプロット
        plt.figure(figsize=(10, 6))
        x = np.arange(module.config.num_classes)
        width = 0.35
        plt.bar(x - width/2, [class_dist[f'class_{i}']['actual'] for i in range(module.config.num_classes)], 
                width, label='Actual')
        plt.bar(x + width/2, [class_dist[f'class_{i}']['predicted'] for i in range(module.config.num_classes)], 
                width, label='Predicted')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(exp_dir, 'class_distribution.png'))
        plt.close()
        
        # 全体のaccuracy
        acc = (all_preds == all_labels).mean()
        print(f'\nOverall Accuracy: {acc:.4f}')
        
        # 結果をJSONとして保存
        results = {
            'accuracy': float(acc),
            'confusion_matrix': cm.tolist(),
            'class_distribution': class_dist,
            'config': module.config.model_dump()
        }
        with open(os.path.join(exp_dir, 'predict_results.json'), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    cli = CLI()
    cli.run()

