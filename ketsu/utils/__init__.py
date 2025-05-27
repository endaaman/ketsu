import os
import glob

from pytorch_lightning.callbacks import EarlyStopping

from .seed import *

class CustomEarlyStopping(EarlyStopping):
    def _improvement_message(self, *args, **kwargs):
        return '\n' + super()._improvement_message(*args, **kwargs)




def resolve_checkpoint_path(path):
    """チェックポイントパスを解決する
    Args:
        path: チェックポイントファイルまたはディレクトリのパス
    Returns:
        str: チェックポイントファイルのパス
    Raises:
        ValueError: チェックポイントが見つからない、または複数見つかった場合
    """
    if os.path.isfile(path):
        return path

    # ディレクトリの場合、最も新しいバージョンを探索
    if os.path.isdir(path):
        # バージョンディレクトリを探索
        version_dirs = sorted(glob.glob(os.path.join(path, 'version_*')))
        if len(version_dirs) > 0:
            return resolve_checkpoint_path(version_dirs[-1])

        ckpts = glob.glob(os.path.join(path, '*.ckpt'))
        if not ckpts:
            raise ValueError(f"No checkpoint files found in {path}")
        if len(ckpts) > 1:
            raise ValueError(f"Multiple checkpoint files found in {path}: {ckpts}")
        return ckpts[0]

    raise ValueError(f"Invalid checkpoint path: {path}")
