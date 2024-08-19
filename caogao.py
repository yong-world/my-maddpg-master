import numpy as np
import tensorflow as tf
from maddpg_pytorch.distributions_pytorch import make_pdtype
from maddpg_pytorch.multi_discrete import MultiDiscrete
import torch.nn.functional as F
import torch
a=[]
for i in range(10000):
    u = torch.rand((2))
    b=-torch.log(-torch.log(u))
    a.append(b)
    print(b)





