import torch
import torch.nn as nn
import torch.nn.functional as F
# 生成标准正态分布，均值为0，标准差为1
def true_distribution(tau_i, z_ij):
    # 标准正态分布：mean=0, std=1
    return torch.normal(mean=0.0, std=1.0, size=z_ij.shape)
# KL散度计算
def kl_divergence(true_dist, approx_dist):
    # 计算KL散度，使用 reduction='batchmean' 进行标准化
    return F.kl_div(approx_dist.log(), true_dist, reduction='batchmean')
# 消息生成模块
class MessageGeneration(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MessageGeneration, self).__init__()
        # 全连接层，用于生成近似分布的均值和方差
        self.fc_mu = nn.Linear(input_dim*3, hidden_dim)
        self.fc_sigma = nn.Linear(input_dim*3, hidden_dim)

    def forward(self, tau_i, z_ij, Q_j_i):
        # 计算近似分布的均值
        mu = self.fc_mu(torch.cat((tau_i, z_ij, Q_j_i), dim=-1))
        # 计算近似分布的方差，并使用exp函数确保方差为正数
        sigma = torch.exp(self.fc_sigma(torch.cat((tau_i, z_ij, Q_j_i), dim=-1)))
        # 生成近似分布样本
        approx_dist = torch.normal(mu, sigma)
        # 生成标准正态分布的真实分布
        true_dist = true_distribution(tau_i, z_ij)
        # 计算KL散度作为损失函数
        kl_loss = kl_divergence(true_dist, approx_dist)
        return kl_loss

# 示例用法
if __name__ == "__main__":
    # 输入维度和隐藏层维度
    input_dim = 10
    hidden_dim = 10
    # 创建消息生成模型
    message_gen = MessageGeneration(input_dim=input_dim, hidden_dim=hidden_dim)
    # 示例输入数据，假设每个输入都是10维度的向量
    tau_i = torch.randn((32, input_dim))  # 批大小为32
    z_ij = torch.randn((32, input_dim))  # 批大小为32
    Q_j_i = torch.randn((32, input_dim))  # 批大小为32
    # 前向传播，计算KL散度损失
    kl_loss = message_gen(tau_i, z_ij, Q_j_i)
    # 打印KL散度损失
    print("KL Divergence Loss: {}".format(kl_loss))
