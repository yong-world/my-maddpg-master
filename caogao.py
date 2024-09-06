import numpy as np
import tensorflow as tf
from maddpg_pytorch.distributions_pytorch import make_pdtype
from maddpg_pytorch.multi_discrete import MultiDiscrete
import torch.nn.functional as F
import torch
import pickle
import pandas
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder部分
        self.encoder_fc1 = nn.Linear(input_dim, 128)
        self.encoder_fc2_mu = nn.Linear(128, latent_dim)  # 均值
        self.encoder_fc2_logvar = nn.Linear(128, latent_dim)  # 方差

        # Decoder部分
        self.decoder_fc1 = nn.Linear(latent_dim, 128)
        self.decoder_fc2 = nn.Linear(128, input_dim)

    def encoder(self, x):
        h1 = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc2_mu(h1)
        logvar = self.encoder_fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h3 = F.relu(self.decoder_fc1(z))
        return torch.sigmoid(self.decoder_fc2(h3))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class Encoder(nn.Module):
    def __init__(self, vae):
        super(Encoder, self).__init__()
        # 直接使用VAE的编码器部分
        self.encoder = vae.encoder

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # 通常，VAE的编码器输出均值和方差，你可以选择使用均值或使用 reparameterize() 得到的 z
        # 这里我们假设使用均值作为输出
        return mu


class CombinedNet(nn.Module):
    def __init__(self, encoder, net2):
        super(CombinedNet, self).__init__()
        self.encoder = encoder
        self.net2 = net2

    def forward(self, x, additional_tensor):
        # 使用编码器部分
        encoded_output,_ = self.encoder(x)

        # 将编码器输出与其他张量合并
        combined = torch.cat((encoded_output, additional_tensor), dim=1)  # 根据需要的维度进行拼接

        # 使用第二个网络
        output2 = self.net2(combined)

        return output2

# 假设你有一个预训练的VAE模型
vae = VAE(input_dim=784, latent_dim=20)  # 假设输入维度为784，隐空间维度为20

# 只使用VAE的编码器部分
encoder = vae.encoder

# 定义你的网络2（例如，一个简单的全连接网络）
net2 = nn.Sequential(
    nn.Linear(20 + 10, 50),  # 假设 additional_tensor 的维度为 10
    nn.ReLU(),
    nn.Linear(50, 10)
)

# 创建组合网络
combined_net = CombinedNet(encoder, net2)

# 输入张量
x = torch.randn(1, 784)  # 示例输入
additional_tensor = torch.randn(1, 10)  # 示例附加张量

# 前向传播
output = combined_net(x, additional_tensor)
