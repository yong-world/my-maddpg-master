import numpy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from maddpg import AgentTrainer
from maddpg_pytorch.replay_buffer import ReplayBuffer
from maddpg_pytorch.distributions_pytorch import make_pdtype
from maddpg_pytorch.vae import VAE, vae_loss
from maddpg_pytorch.information_fusion import MessageEncoder, VariationalApproximation, kl_loss, kl_divergence


def vae_train(trainers):
    vae_obs_n, vae_obs_next_n, vae_act_n, vae_obs, vae_act, vae_rew, vae_obs_next = trainers[
        0].sample_from_replay(trainers)
    loss = []
    n_agents = len(trainers)
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


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, team_num, device='cpu'):
        self.name = name
        self.n = len(obs_shape_n)  # 这里n是智能体个数，obs_shape_n=[34,34,34,34,28,28]
        self.agent_index = agent_index
        self.args = args
        self.device = device
        obs_ph_n = [torch.zeros(obs_shape_n[i].shape).to(self.device) for i in range(self.n)]

        self.vae, self.vae_optimizer = self.init_vae_model_optimizert(
            act_space_n=act_space_n, obs_ph_n=obs_ph_n, agent_index=agent_index,
            lr=args.lr, teammate_num=team_num, num_units=args.num_units, latent_dim=10)

        self.info_fus, self.info_fus_optimizer, self.var_dist, self.var_dist_optimizer = self.init_info_fus(
            act_space_n=act_space_n, obs_ph_n=obs_ph_n, agent_index=agent_index,
            lr=args.lr, latent_dim=10)
        # 初始化模型和优化器
        self.p, self.p_optimizer, self.target_p, self.act_pdtype, self.act_pd = self.init_p_model_optimizer(
            act_space_n=act_space_n,
            obs_ph_n=obs_ph_n, agent_index=agent_index, lr=args.lr, model=model,
            num_units=args.num_units)
        self.q, self.q_optimizer, self.target_q = self.init_q_model_optimizer(
            act_space_n=act_space_n, obs_ph_n=obs_ph_n, lr=args.lr,
            model=model, num_units=args.num_units)

        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def init_info_fus(self, act_space_n, obs_ph_n, agent_index, lr, latent_dim=10):
        z = self.vae.encode_to_z(obs_ph_n[agent_index])
        p_input = len(obs_ph_n[agent_index]) + len(z)
        info_fus = MessageEncoder(p_input, latent_dim).to(self.device)
        var_dist = VariationalApproximation(p_input + 1, latent_dim).to(self.device)
        info_fus_optimizer = torch.optim.Adam(info_fus.parameters(), lr=lr)
        var_dist_optimizer = torch.optim.Adam(var_dist.parameters(), lr=lr)
        return info_fus, info_fus_optimizer, var_dist, var_dist_optimizer

    def init_vae_model_optimizert(self, act_space_n, obs_ph_n, agent_index, teammate_num, num_units, lr, latent_dim=10):
        self_obs_dim = len(obs_ph_n[agent_index])
        act_pdtype = make_pdtype(act_space_n[agent_index])
        self_act_dim = int(act_pdtype.param_shape()[0])
        vae = VAE(input_dim=self_obs_dim, hidden_dim=num_units, latent_dim=latent_dim,
                  obs_output_dim=(teammate_num - 1) * self_obs_dim,
                  act_output_dim=(teammate_num - 1) * self_act_dim).to(self.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
        return vae, optimizer

    def init_p_model_optimizer(self, act_space_n, obs_ph_n, agent_index, model, num_units, lr):
        z = self.vae.encode_to_z(obs_ph_n[agent_index])
        act_pdtype = make_pdtype(act_space_n[agent_index])
        p_input = len(obs_ph_n[agent_index]) + len(z)  # p_index的索引是当前agent的编号 # TODO 优化时改

        # param_shape()是去输出动作的个数的和,因为输出是一个元素的列表所以需要[0]
        p = model(p_input, int(act_pdtype.param_shape()[0]), num_units=num_units).to(self.device)
        act_pd = act_pdtype.pdfromflat(p(torch.rand(p_input).to(self.device)))
        target_p = model(p_input, int(act_pdtype.param_shape()[0]), num_units=num_units).to(self.device)
        p_vars = p.parameters()
        p_optimizer = optim.Adam(p_vars, lr=lr)
        return p, p_optimizer, target_p, act_pdtype, act_pd

    def init_q_model_optimizer(self, act_space_n, obs_ph_n, lr, model, num_units):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # 获取输入的尺寸，包括每个agent的观察和各自的动作空间大小
        q_input = sum(i.numel() for i in obs_ph_n) + sum(i.numel() for i in act_ph_n)
        q = model(q_input, num_outputs=1, num_units=num_units).to(self.device)
        target_q = model(q_input, num_outputs=1, num_units=num_units).to(self.device)
        q_vars = q.parameters()
        q_optimizer = optim.Adam(q_vars, lr=lr)
        return q, q_optimizer, target_q

    def make_update_exp(self, vals, target_vals):
        polyak = 1.0 - 1e-2
        for var, var_target in zip(vals, target_vals):
            var_target.data.copy_(polyak * var_target.data + (1.0 - polyak) * var.data)

    def q_debug(self, obs_n, act_n, target_q_values):
        batch_input = torch.cat([torch.cat(obs_n, dim=1), torch.cat(act_n, dim=1)], dim=1)
        if target_q_values is not False:
            q_val = self.target_q(batch_input)
        else:
            q_val = self.q(batch_input)
        return q_val

    def p_debug(self, obs, target_p_values):
        if target_p_values is not False:
            p_val = self.target_p(obs)
        else:
            p_val = self.p(obs)
        return p_val

    def q_train(self, obs_n, act_n, y, grad_norm_clipping):
        batch_input = torch.cat([torch.cat(obs_n, dim=1), torch.cat(act_n, dim=1)], dim=1)
        q_val = self.q(batch_input.data)

        q_loss = torch.mean(torch.square(q_val - y))
        q_reg = torch.mean(torch.square(q_val))
        loss = q_loss  # + 1e-3 * q_reg
        # print('qtrainloss:{}'.format(q_loss))
        self.q_optimizer.zero_grad()
        loss.backward()

        if grad_norm_clipping is not None:
            for parameter in list(self.q.parameters()):
                torch.nn.utils.clip_grad_norm_(parameter, grad_norm_clipping)  # _原地修改tensor
        self.q_optimizer.step()
        return loss

    def p_train(self, obs_n, act_n, grad_norm_clipping):
        # act_input需要将当前策略根据观察选择的动作的采样放进去，目前梯度裁剪没啥问题
        # 根据智能体当前观察获取当前的动作输出分布采样
        z = self.vae.encode_to_z(obs_n[self.agent_index])
        z2, mu, log_var = self.info_fus(obs_n[self.agent_index], z)
        obs_z2 = torch.cat([obs_n[self.agent_index], z2], -1)

        act_output = self.p(obs_z2)
        act_output_sample = self.act_pd.sample(act_output)
        act_n[self.agent_index] = act_output_sample

        batch_input = torch.cat([torch.cat(obs_n, dim=1), torch.cat(act_n, dim=1)], dim=1)
        q_output = self.q(batch_input)
        q_value_list = q_output.tolist()
        q_value_var = torch.Tensor(q_value_list).to(self.device)
        var_mu, var_log_var = self.var_dist(obs_n[self.agent_index], z, q_value_var)
        kloss=kl_loss(mu1=mu, logvar1=log_var, mu2=var_mu, logvar2=var_log_var)

        q_loss = -torch.mean(q_output)
        p_reg = torch.mean(torch.square(act_output))
        loss = q_loss + p_reg * 1e-3 + kloss

        # print('ptrainqloss:{}\tp_reg:{}'.format(q_loss, p_reg))
        self.p_optimizer.zero_grad()
        self.info_fus_optimizer.zero_grad()
        self.var_dist_optimizer.zero_grad()

        loss.backward()
        if grad_norm_clipping is not None:
            for parameter in list(self.p.parameters()):
                torch.nn.utils.clip_grad_norm_(parameter, grad_norm_clipping)
        self.p_optimizer.step()
        if grad_norm_clipping is not None:
            for parameter in list(self.info_fus.parameters()):
                torch.nn.utils.clip_grad_norm_(parameter, grad_norm_clipping)
        self.info_fus_optimizer.step()
        if grad_norm_clipping is not None:
            for parameter in list(self.var_dist.parameters()):
                torch.nn.utils.clip_grad_norm_(parameter, grad_norm_clipping)
        self.var_dist_optimizer.step()
        return loss

    def action(self, obs):
        # 输入观察输出动作分布的采样
        z = self.vae.encode_to_z(obs)
        z2, mu, log_var = self.info_fus(obs, z)
        obs_z2 = torch.cat([obs, z2], -1)
        flat = self.p(obs_z2)
        return self.act_pd.sample(flat).detach()

    def experience(self, obs, act, rew, new_obs):
        self.replay_buffer.add(obs, act, rew, new_obs)

    def sample_from_replay(self, agents):
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)

        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next = agents[i].replay_buffer.sample_index(index)
            obs_n.append(torch.stack(obs, dim=0))
            obs_next_n.append(torch.stack(obs_next, dim=0))
            act_n.append(torch.stack(act, dim=0))
        # NOTE 取完所有agent的o,a,r,o',done再单独取自己的
        obs, act, rew, obs_next = self.replay_buffer.sample_index(index)
        rew = torch.stack(rew, dim=0)
        return obs_n, obs_next_n, act_n, obs, act, rew, obs_next

    def berman(self, agents, obs_next_n, rew):
        target_q = 0.0

        target_act_next_n = []
        for i in range(self.n):
            z = agents[i].vae.encode_to_z(obs_next_n[i])
            z2, mu, log_var = self.info_fus(obs_next_n[i], z)
            obs_next_z2 = torch.cat([obs_next_n[i], z2], -1)

            flat = agents[i].p_debug(obs_next_z2, target_p_values=True)
            target_act_next_n.append(agents[i].act_pd.sample(flat))
        target_q_next = self.q_debug(obs_next_n, target_act_next_n, target_q_values=True).detach()
        target_q += rew + self.args.gamma * target_q_next
        return target_q

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # 经验池够batch_size*max_episode_len=1024*25=25600
            return
        if t % 100 != 0:  # 每百训练步才更新一次
            return
        # print('p q_training')
        obs_n, obs_next_n, act_n, obs, act, rew, obs_next = self.sample_from_replay(agents)
        target_q = self.berman(agents, obs_next_n, rew)

        q_loss = self.q_train(obs_n=obs_n, act_n=act_n, y=target_q, grad_norm_clipping=0.5)
        p_loss = self.p_train(obs_n=obs_n, act_n=act_n, grad_norm_clipping=0.5)
        # print("q_loss: ", q_loss.data, "p_loss: ", p_loss.data)
        self.make_update_exp(vals=self.p.parameters(), target_vals=self.target_p.parameters())
        self.make_update_exp(vals=self.q.parameters(), target_vals=self.target_q.parameters())
        # 返回值实际并没有被使用
        return True
