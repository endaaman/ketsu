import os
import numpy as np
from pydantic import BaseModel, Field
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Accuracy
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint, LearningRateMonitor


from .utils import BaseCLI, fix_global_seed
from .datasets import ConjDataset
from .models import create_model


class ConjConfig(BaseModel):
    lr: float = 0.0001
    batch_size: int = Field(5, s='-B')
    plateau: bool = False
    nopretrained: bool = False

    with_vessel: bool = Field(False, s='-V')

    arch_name: str = Field('unet16n', l='--arch', s='-A')
    size: int = 512


class CustomEarlyStopping(EarlyStopping):
    def _improvement_message(self, *args, **kwargs):
        return '\n' + super()._improvement_message(*args, **kwargs)

class ConjModule(pl.LightningModule):

    def __init__(self, config:ConjConfig):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config

        self.num_classes = 4 if self.config.with_vessel else 3
        self.unet = create_model(config.arch_name,
                                 num_classes=self.num_classes,
                                 pretrained=not config.nopretrained)
        self.criterion = nn.CrossEntropyLoss()

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
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_jac', jac, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        acc = self.metric_acc(y, t)
        jac = self.metric_jac(y, t)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=False)
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
        if not self.config.plateau:
            return optimizer
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                # verbose=True,
                min_lr=1e-6,
            ),
            'monitor': 'val_loss',   # val_lossを監視
            'interval': 'epoch',
            'frequency': 1
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def on_before_optimizer_step(self, optimizer):
        opt = optimizer
        for param_group in opt.param_groups:
            current_lr = param_group['lr']
            self.log('lr', current_lr, prog_bar=True)

class CLI(BaseCLI):

    class CommonArgs(BaseModel):
        seed: int = 0

    def pre_common(self, a:CommonArgs):
        fix_global_seed(0)
        pl.seed_everything(a.seed)
        torch.set_float32_matmul_precision('medium')
        # matplotlib.use('QtAgg')

    class ModelArgs(CommonArgs):
        arch_name: str = Field('unet16', s='-A', l='--arch')

    def run_model(self, a):
        m = create_model(a.arch_name, num_classes=3)
        t = torch.randn(2, 3, 256, 256)
        print(m(t).shape)

    def run_image(self, a):
        ds = ConjDataset(mode='val', normalization=False)
        image, label = ds[0]
        print(image.shape)
        print(image.dtype)
        print(type(label))
        plt.subplot(1,2,1)
        plt.imshow(Image.fromarray((image.numpy().transpose(1, 2, 0)*255).astype(np.uint8)))
        plt.subplot(1,2,2)
        plt.imshow(label.numpy())
        plt.show()

    class TrainArgs(CommonArgs, ConjConfig):
        num_workers: int = 4
        checkpoint_dir: str = 'checkpoints'
        experiment_name: str = Field('base', l='--exp', s='-E')

    def run_train(self, a:TrainArgs):
        config = ConjConfig(**a.model_dump())

        dir_name = 'v_' + a.arch_name if config.with_vessel else a.arch_name
        checkpoint_dir = os.path.join(a.checkpoint_dir, dir_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_ds = ConjDataset(mode='train', with_vessel=config.with_vessel, augmentation=True)
        val_ds = ConjDataset(mode='val', with_vessel=config.with_vessel, augmentation=False)
        train_loader = DataLoader(train_ds, a.batch_size, num_workers=a.num_workers, shuffle=True)
        val_loader = DataLoader(val_ds, a.batch_size, num_workers=a.num_workers)

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath=checkpoint_dir,
            filename=a.experiment_name or '{version}-{epoch:02d}-{val_loss:.3f}',
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

        logger = TensorBoardLogger(
            save_dir='lightning_logs',
            name=a.experiment_name,
            default_hp_metric=False
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            max_epochs=100,
            devices=1,
            accelerator='gpu',
            benchmark=False,
            callbacks=[RichProgressBar(), checkpoint, early_stopping, lr_monitor],
            log_every_n_steps=1,
            logger=logger,
        )

        print('config', config)
        module = ConjModule(config)
        trainer.fit(module, train_loader, val_loader)

        print(f'\nBest model path: {checkpoint.best_model_path}')

        # Restore best model
        module = ConjModule.load_from_checkpoint(checkpoint.best_model_path)

        test_ds = ConjDataset(mode='val', size=640, with_vessel=config.with_vessel, augmentation=False)
        test_loader = DataLoader(test_ds, a.batch_size, num_workers=a.num_workers)
        # trainer = pl.Trainer(
        #     accelerator='gpu',
        #     devices=1,
        # )
        print(trainer.test(module, test_loader))


    class PredictArgs(CommonArgs):
        checkpoint: str = Field(..., s='-c')
        batch_size: int = Field(16, s='-B')
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
