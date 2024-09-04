import numpy as np
import tensorflow as tf
from maddpg_pytorch.distributions_pytorch import make_pdtype
from maddpg_pytorch.multi_discrete import MultiDiscrete
import torch.nn.functional as F
import torch
import pickle

import sys
import time

for i in range(5):
    time.sleep(2)
    sys.stdout.write("\r now is :{0}".format(i))
    sys.stdout.flush()
# while t<10000:
#     with open('a.pkl', 'ab') as f:
#         pickle.dump(a,f)
#     t+=1
# with open('a.pkl', 'rb') as f:
#     a=pickle.load(f)
#     b=pickle.load(f)
#     print(a,b)

