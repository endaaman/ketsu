from pytorch_lightning.callbacks import EarlyStopping

from .seed import *

class CustomEarlyStopping(EarlyStopping):
    def _improvement_message(self, *args, **kwargs):
        return '\n' + super()._improvement_message(*args, **kwargs)
