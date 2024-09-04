import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, obs_output_dim, act_output_dim):
        super(VAE, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和标准差的参数
        )

        # 解码器部分
        self.decoder_obs = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_output_dim)  # 输出为观察值
        )

        self.decoder_action = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_output_dim)  # 输出为动作分布
        )

    def encode_to_z(self, observation):
        encoded = self.encoder(observation)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)  # 拆分成一半一半，左一半是mu，又一半是log_var对数方差

        # 重参数化技巧：生成隐藏特征 z
        z = self.reparameterize(mu, log_var)
        return z
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码器：输入 -> 隐藏空间参数 (均值和对数方差)
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)  # 拆分成一半一半，左一半是mu，又一半是log_var对数方差

        # 重参数化技巧：生成隐藏特征 z
        z = self.reparameterize(mu, log_var)

        # 解码器：从 z 重构出观察值和动作分布
        recon_obs = self.decoder_obs(z)
        recon_action = self.decoder_action(z)

        return recon_obs, recon_action, mu, log_var


def vae_loss(recon_obs, recon_action, obs, action, mu, log_var):
    # 重构误差 (MSE)
    recon_loss_obs = F.mse_loss(recon_obs, obs, reduction='sum')
    recon_loss_action = F.mse_loss(recon_action, action, reduction='sum')

    # KL 散度
    kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))

    # 总损失
    return recon_loss_obs + recon_loss_action + kl_div
# 假设你已经有obs（观察值）和action（动作）的训练数据
# obs 和 action 都应该是 PyTorch 张量，形状为 [batch_size, input_dim]

# input_dim = obs.shape[1]
# hidden_dim = 128  # 隐藏层维度
# latent_dim = 32  # 隐空间维度
#
# # 创建VAE模型和优化器
# model = VAE(input_dim, hidden_dim, latent_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
# # 训练循环
# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train()
#
#     # 前向传播
#     recon_obs, recon_action, mu, log_var = model(obs)
#
#     # 计算损失
#     loss = vae_loss(recon_obs, recon_action, obs, action, mu, log_var)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 10 == 0:
#         print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
