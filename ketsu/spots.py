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
import pandas as pd
import glob

from .utils import fix_global_seed, CustomEarlyStopping, resolve_checkpoint_path
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



def dump_json(data, save_dir, filename='results.json'):
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {os.path.join(save_dir, filename)}")


class CoralModel(nn.Module):
    def __init__(self, base, num_classes=4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.base = timm.create_model(base, pretrained=pretrained, num_classes=1)
        self.coral_biases = nn.Parameter(torch.arange(0, num_classes-1, dtype=torch.float32).view(1, -1))

    def forward(self, x, with_logits=False):
        raw_logits = self.base(x)
        thresholded_logits = self.coral_biases + raw_logits
        if with_logits:
            return raw_logits, thresholded_logits
        return thresholded_logits


class SpotsConfig(BaseModel):
    fold: int = param(1, s='-f', l='--fold')
    model_name: str = param('resnet34', s='-M', l='--model')
    num_classes: int = 4
    pretrained: bool = True
    lr: float = param(1e-4, s='-l', l='--lr')
    batch_size: int = param(32, s='-B', l='--batch-size')
    max_epochs: int = param(100, s='-e', l='--max-epochs')
    coral_threshold_lr: float = param(1e-4, s='-L', l='--coral-threshold-lr')
    ce_coef: float = param(0.0, s='-c', l='--ce-coef', help='Coefficient for CrossEntropy loss')


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class SpotsLightningModule(pl.LightningModule):
    def __init__(self, config: SpotsConfig):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config

        # モデルの初期化
        self.model = CoralModel(config.model_name, num_classes=config.num_classes, pretrained=config.pretrained)
        self.coral_criterion = nn.BCEWithLogitsLoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.metric_acc = Accuracy(task='multiclass', num_classes=config.num_classes)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _get_prediction(self, logits):
        # CORALの予測をクラスに変換
        # logits: (B, num_classes-1) -> pred: (B,)
        return (logits > 0).sum(dim=1)

    def _get_ce_logits(self, thresholded_logits):
        # CORALのthresholded_logitsからCrossEntropy用のlogitsを生成
        # (B, num_classes-1) -> (B, num_classes)
        batch_size = thresholded_logits.shape[0]
        ce_logits = torch.zeros(batch_size, self.config.num_classes, device=thresholded_logits.device)
        
        # 各クラスのlogitsを計算
        for i in range(self.config.num_classes):
            if i == 0:
                # クラス0: -thresholded_logits[0] (閾値より小さい)
                ce_logits[:, i] = -thresholded_logits[:, 0]
            elif i == self.config.num_classes - 1:
                # 最後のクラス: thresholded_logits[-1] (最後の閾値より大きい)
                ce_logits[:, i] = thresholded_logits[:, -1]
            else:
                # 中間のクラス: thresholded_logits[i] - thresholded_logits[i-1]
                ce_logits[:, i] = thresholded_logits[:, i] - thresholded_logits[:, i-1]
        
        return ce_logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        raw_logits, thresholded_logits = self(x, with_logits=True)
        
        # CORAL loss
        coral_loss = self.coral_criterion(thresholded_logits, y_coral)
        
        # CrossEntropy loss
        ce_logits = self._get_ce_logits(thresholded_logits)
        ce_loss = self.ce_criterion(ce_logits, y)

        # 予測とaccuracyの計算
        pred = self._get_prediction(thresholded_logits)
        y = (y_coral > 0).sum(dim=1)
        acc = self.metric_acc(pred, y)

        # 損失を結合
        loss = coral_loss + self.config.ce_coef * ce_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_coral_loss', coral_loss)
        self.log('train_ce_loss', ce_loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        raw_logits, thresholded_logits = self(x, with_logits=True)
        
        # CORAL loss
        coral_loss = self.coral_criterion(thresholded_logits, y_coral)
        
        # CrossEntropy loss
        ce_logits = self._get_ce_logits(thresholded_logits)
        ce_loss = self.ce_criterion(ce_logits, y)

        # 予測とaccuracyの計算
        pred = self._get_prediction(thresholded_logits)
        y = (y_coral > 0).sum(dim=1)
        acc = self.metric_acc(pred, y)

        # 損失を結合
        loss = coral_loss + self.config.ce_coef * ce_loss

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_coral_loss', coral_loss)
        self.log('val_ce_loss', ce_loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_coral = to_coral_labels(y, num_classes=self.model.num_classes)
        raw_logits, thresholded_logits = self(x, with_logits=True)
        
        # CORAL loss
        coral_loss = self.coral_criterion(thresholded_logits, y_coral)
        
        # CrossEntropy loss
        ce_logits = self._get_ce_logits(thresholded_logits)
        ce_loss = self.ce_criterion(ce_logits, y)

        # 予測とaccuracyの計算
        pred = self._get_prediction(thresholded_logits)
        y = (y_coral > 0).sum(dim=1)
        acc = self.metric_acc(pred, y)

        # 損失を結合
        loss = coral_loss + self.config.ce_coef * ce_loss

        self.log('loss', loss, prog_bar=True)
        self.log('coral_loss', coral_loss)
        self.log('ce_loss', ce_loss)
        self.log('acc', acc, prog_bar=True)
        return {'loss': loss, 'coral_loss': coral_loss, 'ce_loss': ce_loss, 'acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.model.base.parameters(), 'lr': self.config.lr},
            {'params': self.model.coral_biases, 'lr': self.config.coral_threshold_lr}
        ])
        return optimizer


class CLI(AutoCLI):
    class CommonArgs(BaseModel):
        seed: int = 0

    def pre_common(self, a: CommonArgs):
        fix_global_seed(a.seed)
        pl.seed_everything(a.seed)
        torch.set_float32_matmul_precision('medium')

    class TrainArgs(CommonArgs, SpotsConfig):
        num_workers: int = param(4, s='-w', l='--num-workers')
        prefix: str = param('', s='-p', l='--prefix', help='Prefix for experiment name')

    def run_train(self, a: TrainArgs):
        config = SpotsConfig(**a.model_dump())

        # Create experiment name and save directory
        exp_name = a.model_name
        if a.prefix:
            exp_name = f'{a.prefix}_{exp_name}'
        exp_name = f'{exp_name}_fold{a.fold}'
        save_dir = os.path.join('checkpoints', 'spots', exp_name)
        os.makedirs(save_dir, exist_ok=True)

        # Create datasets and dataloaders
        train_dataset = SpotsDataset(fold=a.fold, mode='train', augmentation=True)
        val_dataset = SpotsDataset(fold=a.fold, mode='val', augmentation=False)
        train_loader = DataLoader(train_dataset, a.batch_size, num_workers=a.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, a.batch_size, num_workers=a.num_workers)

        # バッチ数を計算して適切なログ間隔を設定
        num_batches = len(train_loader)
        # エポックあたり約10回程度のログを取るように調整
        log_every_n_steps = max(1, min(100, num_batches // 10))

        # Create logger first to get version
        logger = TensorBoardLogger(os.path.join('checkpoints', 'spots'), name=exp_name, sub_dir='logs')
        version = logger.version
        version_dir = os.path.join(save_dir, f'version_{version}')
        os.makedirs(version_dir, exist_ok=True)

        early_stopping = CustomEarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        )

        # Setup callbacks
        callbacks = [
            early_stopping,
            ModelCheckpoint(
                dirpath=version_dir,
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='{epoch:02d}-{val_loss:.3f}',
            ),
            LearningRateMonitor(logging_interval='epoch'),
            CustomProgressBar()
        ]

        # Create trainer with adjusted logging interval
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices=1,
            log_every_n_steps=log_every_n_steps
        )

        # Train
        module = SpotsLightningModule(config)
        trainer.fit(module, train_loader, val_loader)

        test_ds = SpotsDataset(fold=module.config.fold, mode='val', augmentation=False)
        test_loader = DataLoader(test_ds, a.batch_size, num_workers=a.num_workers)
        test_results = trainer.test(module, test_loader)

        train_ds = SpotsDataset(fold=module.config.fold, mode='train', augmentation=False)
        train_loader = DataLoader(train_ds, a.batch_size, num_workers=a.num_workers)
        train_results = trainer.test(module, train_loader)

        data = {
            'test_results': test_results[0],
            'train_results': train_results[0],
            'coral_biases': module.model.coral_biases.detach().cpu().numpy().tolist()
        }
        dump_json(data, version_dir, 'results.json')
        dump_json(module.config.model_dump(), version_dir, 'config.json')

    class ModelArgs(CommonArgs):
        model_name: str = param('resnet34', s='-M', l='--model')

    def run_model(self, a: ModelArgs):
        model = CoralModel(a.model_name, num_classes=4, pretrained=False)
        t = torch.randn(2, 3, 256, 256)
        print(model(t).shape)

    class TestArgs(CommonArgs):
        checkpoint: str = param(..., s='-C')
        batch_size: int = param(32, s='-B')
        num_workers: int = 4

    def run_test(self, a: TestArgs):
        module = SpotsLightningModule.load_from_checkpoint(a.checkpoint)
        test_dataset = SpotsDataset(fold=module.config.fold, mode='val', augmentation=False)
        test_loader = DataLoader(test_dataset, a.batch_size, num_workers=a.num_workers)

        trainer = pl.Trainer(accelerator='auto', devices=1)
        results = trainer.test(module, test_loader)
        print(results)

    class PredictArgs(CommonArgs):
        checkpoint: str = param(..., s='-C')
        batch_size: int = param(32, s='-B')
        num_workers: int = 4
        save: bool = False

    def run_predict(self, a: PredictArgs):
        checkpoint_path = resolve_checkpoint_path(a.checkpoint)
        module = SpotsLightningModule.load_from_checkpoint(checkpoint_path)
        print('Config:', module.config)

        # デバイスの設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        module = module.to(device)
        module.eval()

        # チェックポイントのパスから実験ディレクトリを取得
        exp_dir = os.path.dirname(checkpoint_path)
        version_dir = os.path.dirname(exp_dir)  # version_0 の親ディレクトリ
        version = os.path.basename(exp_dir).split('_')[1]  # version_0 から 0 を取得

        test_dataset = SpotsDataset(fold=module.config.fold, mode='val', augmentation=False)
        test_loader = DataLoader(test_dataset, a.batch_size, num_workers=a.num_workers)

        # 予測を収集
        all_preds = []
        all_labels = []
        all_raw_logits = []
        all_thresholded_logits = []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                raw_logits, thresholded_logits = module(x, with_logits=True)
                preds = module._get_prediction(thresholded_logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_raw_logits.extend(raw_logits.cpu().numpy())
                all_thresholded_logits.extend(thresholded_logits.cpu().numpy())

        # 結果を表示
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_raw_logits = np.array(all_raw_logits)
        all_thresholded_logits = np.array(all_thresholded_logits)

        # logitsをCSVに保存
        logits_df = pd.DataFrame(all_raw_logits, columns=[f'raw_logit_{i}' for i in range(all_raw_logits.shape[1])])
        thresholded_df = pd.DataFrame(all_thresholded_logits, columns=[f'thresholded_logit_{i}' for i in range(all_thresholded_logits.shape[1])])
        logits_df = pd.concat([logits_df, thresholded_df], axis=1)
        logits_df['filename'] = test_dataset.df['filename'].tolist()
        logits_df['predicted_class'] = all_preds
        logits_df['true_class'] = all_labels
        logits_df.to_csv(os.path.join(exp_dir, 'logits.csv'), index=False)
        print(f'\nLogits saved to: {os.path.join(exp_dir, "logits.csv")}')

        # coral_biasesを取得
        biases = module.model.coral_biases.detach().cpu().numpy().flatten()
        print(f'\nCORAL biases: {biases}')

        # 生のlogits分布
        plt.figure(figsize=(15, 5))
        for i in range(all_raw_logits.shape[1]):
            plt.subplot(1, all_raw_logits.shape[1], i+1)
            sns.histplot(data=logits_df, x=f'raw_logit_{i}', hue='true_class', multiple='stack')
            plt.axvline(x=-biases[i], color='r', linestyle='--', label=f'Threshold ({-biases[i]:.2f})')
            plt.title(f'Raw Logit {i} Distribution')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'raw_logits_histogram.png'))
        plt.close()

        # クラスごとの生のlogits分布
        plt.figure(figsize=(15, 5))
        for i in range(all_raw_logits.shape[1]):
            plt.subplot(1, all_raw_logits.shape[1], i+1)
            sns.boxplot(data=logits_df, x='true_class', y=f'raw_logit_{i}')
            plt.axhline(y=-biases[i], color='r', linestyle='--', label=f'Threshold ({-biases[i]:.2f})')
            plt.title(f'Raw Logit {i} by Class')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'raw_logits_by_class.png'))
        plt.close()

        # 閾値調整後のlogits分布
        plt.figure(figsize=(15, 5))
        for i in range(all_raw_logits.shape[1]):
            plt.subplot(1, all_raw_logits.shape[1], i+1)
            adjusted_logits = logits_df[f'raw_logit_{i}'] + biases[i]
            sns.histplot(x=adjusted_logits, hue=logits_df['true_class'], multiple='stack')
            plt.axvline(x=0, color='r', linestyle='--', label='Decision Boundary')
            plt.title(f'Adjusted Logit {i} Distribution')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'adjusted_logits_histogram.png'))
        plt.close()

        # クラスごとの調整後logits分布
        plt.figure(figsize=(15, 5))
        for i in range(all_raw_logits.shape[1]):
            plt.subplot(1, all_raw_logits.shape[1], i+1)
            adjusted_logits = logits_df[f'raw_logit_{i}'] + biases[i]
            sns.boxplot(x=logits_df['true_class'], y=adjusted_logits)
            plt.axhline(y=0, color='r', linestyle='--', label='Decision Boundary')
            plt.title(f'Adjusted Logit {i} by Class')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'adjusted_logits_by_class.png'))
        plt.close()

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

        results = {
            'accuracy': float(acc),
            'confusion_matrix': cm.tolist(),
            'class_distribution': class_dist,
            'coral_biases': module.model.coral_biases.detach().cpu().numpy().tolist()
        }
        dump_json(results, exp_dir, 'predict_results.json')

    class AggregateArgs(CommonArgs):
        model_name: str = param('resnet34', s='-M', l='--model')
        prefix: str = param('', s='-p', l='--prefix', help='Prefix for experiment name')

    def run_aggregate(self, a: AggregateArgs):
        """全foldの結果を集計する"""
        base_dir = os.path.join('checkpoints', 'spots')
        exp_name = a.model_name
        if a.prefix:
            exp_name = f'{a.prefix}_{exp_name}'
        report_dir = os.path.join('reports', 'spots', exp_name)
        os.makedirs(report_dir, exist_ok=True)

        # 各foldの結果を収集
        all_logits = []
        all_preds = []
        all_labels = []
        fold_results = []
        fold_sizes = []  # 各foldのデータ数を記録
        all_biases = []  # 各foldのcoral_biasesを保存

        for fold in range(1, 6):  # fold 1-5
            fold_exp_name = f'{exp_name}_fold{fold}'
            exp_dir = os.path.join(base_dir, fold_exp_name)
            
            try:
                # 最新のバージョンディレクトリを取得
                version_dirs = sorted(glob.glob(os.path.join(exp_dir, 'version_*')))
                if not version_dirs:
                    print(f"Warning: No version directories found for {fold_exp_name}")
                    continue
                latest_version = version_dirs[-1]

                # チェックポイントを読み込み
                ckpts = glob.glob(os.path.join(latest_version, '*.ckpt'))
                if not ckpts:
                    print(f"Warning: No checkpoint found in {latest_version}")
                    continue
                if len(ckpts) > 1:
                    print(f"Warning: Multiple checkpoints found in {latest_version}, using the first one")
                checkpoint_path = ckpts[0]

                # coral_biasesをJSONから読み込み
                if a.model_type == 'coral':
                    results_json = os.path.join(latest_version, 'results.json')
                    if os.path.exists(results_json):
                        with open(results_json, 'r') as f:
                            results = json.load(f)
                            biases = np.array(results['coral_biases']).flatten()  # 1次元に変換
                            all_biases.append(biases)
                            print(f"\nFold {fold} coral_biases: {biases}")

                # logits.csvを読み込み
                logits_path = os.path.join(latest_version, 'logits.csv')
                if not os.path.exists(logits_path):
                    print(f"Warning: No logits.csv found in {latest_version}")
                    continue

                # CSVを読み込み
                df = pd.read_csv(logits_path)
                fold_sizes.append(len(df))

                # 結果を保存
                if a.model_type == 'coral':
                    logit_cols = [col for col in df.columns if col.startswith('raw_logit_')]
                    all_logits.extend(df[logit_cols].values)
                all_preds.extend(df['predicted_class'].values)
                all_labels.extend(df['true_class'].values)

                # foldごとの結果を記録
                acc = (df['predicted_class'] == df['true_class']).mean()
                fold_results.append({
                    'fold': fold,
                    'accuracy': float(acc),
                    'checkpoint': checkpoint_path,
                    'num_samples': len(df)
                })

            except Exception as e:
                print(f"Error processing fold {fold}: {e}")
                continue

        # 結果をCSVに保存
        if a.model_type == 'coral':
            logits_df = pd.DataFrame(all_logits, columns=[f'raw_logit_{i}' for i in range(all_logits[0].shape[0])])
            # foldの列を正しく設定
            fold_column = []
            for fold, size in enumerate(fold_sizes, 1):
                fold_column.extend([fold] * size)
            logits_df['fold'] = fold_column
            logits_df['predicted_class'] = all_preds
            logits_df['true_class'] = all_labels
            logits_df.to_csv(os.path.join(report_dir, 'logits.csv'), index=False)

            # coral_biasesの分布をプロット
            plt.figure(figsize=(15, 5))
            biases_df = pd.DataFrame(all_biases, columns=[f'bias_{i}' for i in range(len(all_biases[0]))])
            sns.boxplot(data=biases_df)
            plt.title('Distribution of CORAL Biases across Folds')
            plt.xlabel('Threshold Index')
            plt.ylabel('Bias Value')
            plt.savefig(os.path.join(report_dir, 'coral_biases_distribution.png'))
            plt.close()

            # 全体の生のlogits分布
            plt.figure(figsize=(15, 5))
            for i in range(all_logits[0].shape[0]):
                plt.subplot(1, all_logits[0].shape[0], i+1)
                sns.histplot(data=logits_df, x=f'raw_logit_{i}', hue='true_class', multiple='stack')
                plt.title(f'Raw Logit {i} Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'raw_logits_histogram.png'))
            plt.close()

            # 全体のクラスごとの生のlogits分布
            plt.figure(figsize=(15, 5))
            for i in range(all_logits[0].shape[0]):
                plt.subplot(1, all_logits[0].shape[0], i+1)
                sns.boxplot(data=logits_df, x='true_class', y=f'raw_logit_{i}')
                plt.title(f'Raw Logit {i} by Class')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'raw_logits_by_class.png'))
            plt.close()

        # 各foldの結果をCSVに保存
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(os.path.join(report_dir, 'results.csv'), index=False)

        # 混同行列
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'))
        plt.close()

        # クラス分布
        plt.figure(figsize=(10, 6))
        x = np.arange(4)  # クラス数は4固定
        width = 0.35
        plt.bar(x - width/2, [np.sum(np.array(all_labels) == i) for i in range(4)],
                width, label='Actual')
        plt.bar(x + width/2, [np.sum(np.array(all_preds) == i) for i in range(4)],
                width, label='Predicted')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(report_dir, 'class_distribution.png'))
        plt.close()

        print(f"\nResults saved to: {report_dir}")
        print(f"Overall accuracy: {np.mean([r['accuracy'] for r in fold_results]):.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
        print("\nFold details:")
        for r in fold_results:
            print(f"Fold {r['fold']}: accuracy={r['accuracy']:.4f}, samples={r['num_samples']}")



if __name__ == '__main__':
    cli = CLI()
    cli.run()

