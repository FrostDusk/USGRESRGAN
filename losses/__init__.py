import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import LOSS_REGISTRY

# Explicitly import GAN-related functions ONLY
from .gan_loss import g_path_regularize, gradient_penalty_loss, r1_penalty

__all__ = ['build_loss', 'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize']

# Scan all loss modules except gan_loss.py to avoid re-registration
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(loss_folder)
    if v.endswith('_loss.py') and 'gan_loss' not in v
]

# Dynamically import remaining loss modules
_model_modules = [
    importlib.import_module(f'basicsr.losses.{file_name}') for file_name in loss_filenames
]

def build_loss(opt):
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
