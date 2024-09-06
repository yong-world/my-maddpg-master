import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, obs_output_dim, act_output_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.z_dim = latent_dim
        self.obs_output_dim = obs_output_dim
        self.act_output_dim = act_output_dim
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



def vae_train(trainers):
    vae_obs_n, vae_obs_next_n, vae_act_n, vae_obs, vae_act, vae_rew, vae_obs_next = trainers[
        0].sample_from_replay(trainers)
    loss = []
    n_agents=len(trainers)
    for i in range(n_agents):
        teammate_obs = [vae_obs_n[j] for j in range(n_agents) if j != i]
        teammate_obs = torch.cat(teammate_obs, -1)
        teammate_act = [vae_act_n[j] for j in range(n_agents) if j != i]
        teammate_act = torch.cat(teammate_act, -1)
        recon_obs, recon_action, mu, log_var = trainers[i].vae(vae_obs_n[i])
        agent_i_vae_loss = vae_loss(recon_obs=recon_obs, recon_action=recon_action, obs=teammate_obs,
                                    action=teammate_act, mu=mu, log_var=log_var)
        loss.append(agent_i_vae_loss.item())
        trainers[i].vae_optimizer.zero_grad()
        agent_i_vae_loss.backward()
        for parameter in list(trainers[i].vae.parameters()):
            torch.nn.utils.clip_grad_norm_(parameter, 0.5)
        trainers[i].vae_optimizer.step()
    return loss
