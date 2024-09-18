import numpy as np
import tensorflow as tf
from maddpg_pytorch.distributions_pytorch import make_pdtype
from maddpg_pytorch.multi_discrete import MultiDiscrete
import torch.nn.functional as F
import torch
import pickle
# Import the summary writer
# from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object

from tensorboardX import SummaryWriter
writer = SummaryWriter()
for i in range(100):
    writer.add_scalar('train/loss', i % 5, i)
