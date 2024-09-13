import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


# 1. 消息生成器模块（Message Encoder）
class MessageEncoder(nn.Module):
    def __init__(self, input_dim, message_dim):
        super(MessageEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, 2 * message_dim)  # 一层全连接层，输出均值和对数方差

    def forward(self, obs, z):
        input = torch.cat((obs, z), dim=-1)
        h = self.fc(input)
        mu, logvar = h.chunk(2, dim=-1)  # 将输出拆分成均值和对数方差
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 采样
        message = mu + eps * std  # 重参数化技巧
        return message, mu, logvar


# 2. 变分分布近似（Variational Distribution Approximation）
class VariationalApproximation(nn.Module):
    def __init__(self, input_dim, message_dim):
        super(VariationalApproximation, self).__init__()
        self.fc = nn.Linear(input_dim, 2 * message_dim)  # 一层全连接层，输出均值和对数方差

    def forward(self, obs, z, q):
        input = torch.cat([obs, z, q], dim=-1)
        h = self.fc(input)
        mu, logvar = h.chunk(2, dim=-1)  # 将输出拆分成均值和对数方差
        return mu, logvar


# 3. KL 散度计算
def kl_loss(mu1, logvar1, mu2, logvar2):
    dist1 = Normal(mu1, torch.exp(0.5 * logvar1))  # p分布
    dist2 = Normal(mu2, torch.exp(0.5 * logvar2))  # qξ分布
    return kl_divergence(dist1, dist2).mean()  # KL散度的平均


# 4. 损失函数，包括KL散度和其他任务损失
def loss_fn(recon_loss, mu1, logvar1, mu2, logvar2):
    kl = kl_loss(mu1, logvar1, mu2, logvar2)  # 计算KL散度
    total_loss = recon_loss + kl  # 总损失为任务损失和KL散度的加权和
    return total_loss, kl

# 6. 数据准备和模型训练
# input_dim = 32  # 输入维度
# message_dim = 16  # 消息维度
# message_encoder = MessageEncoder(input_dim, message_dim)
# optimizer = optim.Adam(message_encoder.parameters(), lr=0.01)
# var_dist=VariationalApproximation(input_dim,message_dim)
# message, mu1, logvar1 = message_encoder()
# q_mu, q_logvar = var_dist()
