import torch
import torch.nn as nn
import torch.nn.functional as F

# 通常のクロスエントロピーロス（すでに始められているものを完成）
class CrossEntropy(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, x, y):
        # x: [B, C, H, W] - 予測logits
        # y: [B, H, W] - ターゲット（整数クラスラベル）
        return F.cross_entropy(
            x, y,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean'
        )

# Dice Loss - セグメンテーションによく使われる
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, x, y):
        # x: [B, C, H, W] - 予測logits
        # y: [B, H, W] - ターゲット

        # softmaxでクラス確率に変換
        x_softmax = F.softmax(x, dim=1)

        # one-hotエンコーディング
        batch_size, num_classes = x.size(0), x.size(1)
        y_onehot = F.one_hot(y, num_classes).permute(0, 3, 1, 2).float()

        # インターセクションとユニオンを計算
        intersection = torch.sum(x_softmax * y_onehot, dim=[0, 2, 3])
        union = torch.sum(x_softmax, dim=[0, 2, 3]) + torch.sum(y_onehot, dim=[0, 2, 3])

        # Diceスコアを計算
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)

        # クラスごとに計算されたDiceスコアの平均を取る
        dice_loss = 1.0 - torch.mean(dice)

        return dice_loss

# Focal Loss - クラス不均衡問題に対応
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, x, y):
        # x: [B, C, H, W] - 予測logits
        # y: [B, H, W] - ターゲット

        # CEロス計算の準備
        log_softmax = F.log_softmax(x, dim=1)

        # ピクセルごとのクラス確率を抽出
        batch_size, num_classes, height, width = x.shape
        y_flat = y.view(-1)

        # ignore_indexに対応
        valid_mask = y_flat != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        y_flat = y_flat[valid_mask]
        log_probs = log_softmax.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        log_probs = log_probs[valid_mask]

        # CEロス（ターゲットクラスの負の対数確率）
        ce_loss = F.nll_loss(log_probs, y_flat, weight=self.weight, reduction='none')

        # pt計算（正しいクラスの予測確率）
        pt = torch.exp(-ce_loss)

        # Focal Lossの計算
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

# IoU Loss - Intersection over Union（IoU）に基づくロス
class IoULoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, x, y):
        # x: [B, C, H, W] - 予測logits
        # y: [B, H, W] - ターゲット

        # softmaxでクラス確率に変換
        x_softmax = F.softmax(x, dim=1)

        # one-hotエンコーディング
        batch_size, num_classes = x.size(0), x.size(1)
        y_onehot = F.one_hot(y, num_classes).permute(0, 3, 1, 2).float()

        # インターセクションとユニオンを計算
        intersection = torch.sum(x_softmax * y_onehot, dim=[0, 2, 3])
        union = torch.sum(x_softmax, dim=[0, 2, 3]) + torch.sum(y_onehot, dim=[0, 2, 3]) - intersection

        # IoUスコアを計算
        iou = (intersection + self.smooth) / (union + self.smooth + self.eps)

        # クラスごとに計算されたIoUスコアの平均を取る
        iou_loss = 1.0 - torch.mean(iou)

        return iou_loss

# 複合損失（CrossEntropyとDiceLossの組み合わせ）
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, weight=None, ignore_index=-100):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropy(weight=weight, ignore_index=ignore_index)
        self.dice_loss = DiceLoss()

    def forward(self, x, y):
        ce_loss = self.ce_loss(x, y)
        dice_loss = self.dice_loss(x, y)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_loss(loss_name):
    loss_name = loss_name.lower()

    if loss_name == 'ce' or loss_name == 'cross_entropy':
        return CrossEntropy()

    if loss_name == 'dice':
        return DiceLoss()

    if loss_name == 'focal':
        return FocalLoss()

    if loss_name == 'iou':
        return IoULoss()

    if loss_name == 'combined':
        return CombinedLoss()

    raise ValueError(f"未知のロス関数名: {loss_name}")
