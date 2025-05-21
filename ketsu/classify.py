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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        logits = self(x)
        loss = self.criterion(logits, y_coral)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        logits = self(x)
        loss = self.criterion(logits, y_coral)
        self.log('val_loss', loss, prog_bar=True)
        return loss

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
        fold: int = param(1, s='-f', l='--fold')
        num_workers: int = param(4, s='-w', l='--num-workers')
        experiment_name: str = param('base', l='--exp', s='-E')

    def run_train(self, a: TrainArgs):
        config = CoralConfig(**a.model_dump())

        # Create experiment name and save directory
        exp_name = f'coral_{a.model_name}_fold{a.fold}'
        if a.experiment_name != 'base':
            exp_name = f'{exp_name}_{a.experiment_name}'
        save_dir = os.path.join('checkpoints', exp_name)
        os.makedirs(save_dir, exist_ok=True)

        # Create datasets and dataloaders
        train_dataset = SpotsDataset(fold=a.fold, mode='train', augmentation=True)
        val_dataset = SpotsDataset(fold=a.fold, mode='val', augmentation=False)
        train_loader = DataLoader(train_dataset, a.batch_size, num_workers=a.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, a.batch_size, num_workers=a.num_workers)

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, mode='min'),
            ModelCheckpoint(
                dirpath=save_dir,
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
            logger=TensorBoardLogger('checkpoints', name=exp_name),
            accelerator='auto',
            devices=1
        )

        # Train
        module = CoralLightningModule(config)
        trainer.fit(module, train_loader, val_loader)

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
        print(module.config)

        test_dataset = SpotsDataset(fold=module.config.fold, mode='val', augmentation=False)
        test_loader = DataLoader(test_dataset, a.batch_size, num_workers=a.num_workers)

        trainer = pl.Trainer(accelerator='auto', devices=1)
        results = trainer.test(module, test_loader)
        print(results)


if __name__ == '__main__':
    cli = CLI()
    cli.run()

