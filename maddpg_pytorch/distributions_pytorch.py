import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal, Bernoulli
from gym import spaces
# from maddpg_pytorch.multi_discrete import MultiDiscrete
from multiagent.multi_discrete import MultiDiscrete

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def logp(self, x):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return torch.zeros(prepend_shape + self.param_shape(), dtype=torch.float32)

    def sample_placeholder(self, prepend_shape, name=None):
        return torch.zeros(self.sample_shape(), dtype=self.sample_dtype())

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return torch.int32

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat  # 感觉ncat就是num of categorical

    def pdclass(self):
        return SoftCategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return [self.ncat]

    def sample_dtype(self):
        return torch.float32

class MultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1

    def pdclass(self):
        return MultiCategoricalPd

    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.low, self.high, flat)

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return torch.int32

class SoftMultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1

    def pdclass(self):
        return SoftMultiCategoricalPd

    def pdfromflat(self, flat):
        return SoftMultiCategoricalPd(self.low, self.high, flat)

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [sum(self.ncats)]

    def sample_dtype(self):
        return torch.float32

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return torch.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BernoulliPd

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return torch.int32

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return torch.argmax(self.logits, dim=1)

    def logp(self, x):
        return -F.cross_entropy(self.logits, x, reduction='none')

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True).values
        a1 = other.logits - torch.max(other.logits, dim=1, keepdim=True).values
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = ea0.sum(dim=1, keepdim=True)
        z1 = ea1.sum(dim=1, keepdim=True)
        p0 = ea0 / z0
        return (p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))).sum(dim=1)

    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True).values
        ea0 = torch.exp(a0)
        z0 = ea0.sum(dim=1, keepdim=True)
        p0 = ea0 / z0
        return (p0 * (torch.log(z0) - a0)).sum(dim=1)

    def sample(self):
        u = torch.rand_like(self.logits)
        return torch.argmax(self.logits - torch.log(-torch.log(u)), dim=1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftCategoricalPd(Pd):
    # 使用Gumbel-softmax实现的软分类分布
    # 通过对网络输出增加Gumbel噪声达到类似于采样的效果同时保持可微性
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits
    def set_flat(self, new_flat):
        self.logits = new_flat
    def mode(self):
        return F.softmax(self.logits, dim=-1)

    def logp(self, x):
        return -F.cross_entropy(self.logits, x, reduction='none')

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True).values
        a1 = other.logits - torch.max(other.logits, dim=1, keepdim=True).values
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = ea0.sum(dim=1, keepdim=True)
        z1 = ea1.sum(dim=1, keepdim=True)
        p0 = ea0 / z0
        return (p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))).sum(dim=1)

    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True).values
        ea0 = torch.exp(a0)
        z0 = ea0.sum(dim=1, keepdim=True)
        p0 = ea0 / z0
        return (p0 * (torch.log(z0) - a0)).sum(dim=1)

    def sample(self):
        u = torch.rand_like(self.logits)
        return F.softmax(self.logits - torch.log(-torch.log(u)), dim=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(Pd):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = torch.tensor(low, dtype=torch.int32)
        self.categoricals = [CategoricalPd(logits) for logits in torch.split(flat, high - low + 1, dim=len(flat.shape) - 1)]

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.low + torch.stack([p.mode() for p in self.categoricals], dim=-1).to(torch.int32)

    def logp(self, x):
        return sum(p.logp(px) for p, px in zip(self.categoricals, torch.unbind(x - self.low, dim=len(x.shape) - 1)))

    def kl(self, other):
        return sum(p.kl(q) for p, q in zip(self.categoricals, other.categoricals))

    def entropy(self):
        return sum(p.entropy() for p in self.categoricals)

    def sample(self):
        return self.low + torch.stack([p.sample() for p in self.categoricals], dim=-1).to(torch.int32)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftMultiCategoricalPd(Pd):  # doesn't work yet不，还在用
    def __init__(self, low, high, flat):
        # flat是当前actor网络输出，high其中的值分别对应4和3，low是0
        self.flat = flat
        self.low = torch.tensor(low, dtype=torch.float32)
        # 将当前actor网络输出(flat)拆分成[[5个]，[4个]]然后去实例化单个的软分类分布
        # self.categoricals = [SoftCategoricalPd(logits) for logits in torch.split(flat, high - low + 1, dim=len(flat.shape) - 1)]
        self.categoricals = [SoftCategoricalPd(logits) for logits in
                             torch.split(flat, tuple(high - low + 1), dim=0)]

    def flatparam(self):
        return self.flat
    def set_flat(self, new_flat):
        self.flat = new_flat
    def mode(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].mode())
        return torch.cat(x, dim=-1)

    def logp(self, x):
        return sum(p.logp(px) for p, px in zip(self.categoricals, torch.unbind(x - self.low, dim=len(x.shape) - 1)))

    def kl(self, other):
        return sum(p.kl(q) for p, q in zip(self.categoricals, other.categoricals))

    def entropy(self):
        return sum(p.entropy() for p in self.categoricals)

    def sample(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].sample())
        return torch.cat(x, dim=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = torch.chunk(flat, 2, dim=len(flat.shape) - 1)
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def logp(self, x):
        return -0.5 * torch.sum(((x - self.mean) / self.std)**2 + 2 * self.logstd + np.log(2.0 * np.pi), dim=len(x.shape) - 1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return torch.sum(other.logstd - self.logstd + (self.std**2 + (self.mean - other.mean)**2) / (2.0 * other.std**2) - 0.5, dim=len(self.mean.shape) - 1)

    def entropy(self):
        return torch.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), dim=len(self.mean.shape) - 1)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return torch.round(torch.sigmoid(self.logits))

    def logp(self, x):
        return -F.binary_cross_entropy_with_logits(self.logits, x, reduction='none').sum(dim=len(x.shape) - 1)

    def kl(self, other):
        assert isinstance(other, BernoulliPd)
        return torch.sum(torch.sigmoid(self.logits) * (torch.log(torch.sigmoid(self.logits)) - torch.log(torch.sigmoid(other.logits))) + (1 - torch.sigmoid(self.logits)) * (torch.log(1 - torch.sigmoid(self.logits)) - torch.log(1 - torch.sigmoid(other.logits))), dim=len(self.logits.shape) - 1)

    def entropy(self):
        return torch.sum(F.binary_cross_entropy_with_logits(self.logits, torch.sigmoid(self.logits), reduction='none'), dim=len(self.logits.shape) - 1)

    def sample(self):
        return Bernoulli(logits=self.logits).sample()

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        # return CategoricalPdType(ac_space.n)
        return SoftCategoricalPdType(ac_space.n)
    elif isinstance(ac_space, MultiDiscrete):
        # return MultiCategoricalPdType(ac_space.low, ac_space.high)
        return SoftMultiCategoricalPdType(ac_space.low, ac_space.high)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError

def shape_el(v, i):
    maybe = v.shape[i]
    if maybe is not None:
        return maybe
    else:
        return torch.tensor(v.size(i), dtype=torch.int32)
