import numpy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from maddpg import AgentTrainer
from maddpg_pytorch.replay_buffer import ReplayBuffer
from maddpg_pytorch.distributions_pytorch import make_pdtype


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = [torch.zeros(obs_shape_n[i]) for i in range(self.n)]

        # 初始化模型和优化器
        self.p, self.p_optimizer, self.target_p, self.target_p_optimizer, self.act_pdtype, = self.init_p_model_optimizer(
            act_space_n=act_space_n,
            obs_ph_n=obs_ph_n, agent_index=agent_index, lr=args.lr, model=model,
            num_units=args.num_units)
        self.q, self.q_optimizer, self.target_q, self.target_q_optimizer = self.init_q_model_optimizer(
            act_space_n=act_space_n, obs_ph_n=obs_ph_n, lr=args.lr,
            model=model, num_units=args.num_units)

        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def make_update_exp(self, vals, target_vals):
        polyak = 1.0 - 1e-2
        for var, var_target in zip(vals, target_vals):
            var_target.data.copy_(polyak * var_target.data + (1.0 - polyak) * var.data)

    def q_debug(self, obs_n, act_n, target_q_values):
        batch_input = np.concatenate([np.concatenate(obs_n, axis=1), np.concatenate(act_n, axis=1)], axis=1)
        batch_input = torch.from_numpy(batch_input)
        if target_q_values is not False:
            q_val = self.target_q(batch_input)
        else:
            q_val = self.q(batch_input)
        return q_val

    def p_debug(self, obs, target_p_values):
        batch_input = torch.from_numpy(obs)
        if target_p_values is not False:
            p_val = self.target_p(batch_input)
        else:
            p_val = self.p(batch_input)
        return p_val

    def q_train(self, obs_n, act_n, y, grad_norm_clipping):
        batch_input = np.concatenate([np.concatenate(obs_n, axis=1), np.concatenate(act_n, axis=1)], axis=1)
        batch_input = torch.from_numpy(batch_input)
        y = torch.from_numpy(y)
        q_val = self.q(batch_input)
        q_loss = torch.mean(torch.square(y - q_val))
        q_reg = torch.mean(torch.square(q_val))
        loss = q_loss  # + 1e-3 * q_reg
        self.q_optimizer.zero_grad()
        loss.backward()

        if grad_norm_clipping is not None:
            for parameter in list(self.p.parameters()):
                torch.nn.utils.clip_grad_norm_(parameter, grad_norm_clipping)  # _原地修改tensor
        self.q_optimizer.step()
        return loss

    def p_train(self, obs_n, act_n, grad_norm_clipping):
        # TODO act_input需要将当前策略根据观察选择的动作的采样放进去，目前梯度裁剪没啥问题
        obs_n = [torch.from_numpy(obs_i) for obs_i in obs_n]
        act_n = [torch.from_numpy(act_i) for act_i in act_n]
        # 根据智能体当前观察获取当前的动作输出分布采样
        act_output = self.p(obs_n[self.agent_index])
        act_output_sample = torch.empty(act_output.shape)
        for i in range(act_output.shape[0]):
            act_output_sample[i] = self.act_pdtype.pdfromflat(act_output[i]).sample()
        act_n[self.agent_index] = act_output_sample
        #
        batch_input = torch.cat([torch.cat(obs_n, dim=1), torch.cat(act_n, dim=1)], dim=1)
        q_loss = -torch.mean(self.q(batch_input))
        p_reg = torch.mean(torch.square(act_output))
        loss = q_loss + p_reg * 1e-3
        self.p_optimizer.zero_grad()
        loss.backward()
        if grad_norm_clipping is not None:
            for parameter in list(self.p.parameters()):
                torch.nn.utils.clip_grad_norm_(parameter, grad_norm_clipping)
        self.p_optimizer.step()
        return loss

    def init_p_model_optimizer(self, act_space_n, obs_ph_n, agent_index, model, num_units, lr):
        # TODO act_pdtype_n转成act_pdtype会快点吧
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        p_input = len(obs_ph_n[agent_index])  # p_index的索引是当前agent的编号
        # param_shape()是去输出动作的个数的和,因为输出是一个元素的列表所以需要[0]
        p = model(p_input, int(act_pdtype_n[agent_index].param_shape()[0]), num_units=num_units)
        target_p = model(p_input, int(act_pdtype_n[agent_index].param_shape()[0]), num_units=num_units)
        p_vars = list(p.parameters())
        target_p_vars = list(target_p.parameters())
        p_optimizer = optim.Adam(p_vars, lr=lr)
        target_p_optimizer = optim.Adam(target_p_vars, lr=lr)
        return p, p_optimizer, target_p, target_p_optimizer, act_pdtype_n[agent_index]

    def init_q_model_optimizer(self, act_space_n, obs_ph_n, lr, model, num_units):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # TODO act_ph_n改进，ptrain同理
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # 获取输入的尺寸，包括每个agent的观察和各自的动作空间大小
        q_input = sum(i.numel() for i in obs_ph_n) + sum(i.numel() for i in act_ph_n)
        q = model(q_input, num_outputs=1, num_units=num_units)
        target_q = model(q_input, num_outputs=1, num_units=num_units)
        q_vars = list(q.parameters())
        target_q_vars = list(target_q.parameters())
        q_optimizer = optim.Adam(q_vars, lr=lr)
        target_q_optimizer = optim.Adam(target_q_vars, lr=lr)
        return q, q_optimizer, target_q, target_q_optimizer

    def action(self, obs):
        # 输入观察输出动作分布的采样
        flat = self.p(torch.from_numpy(obs).to(dtype=torch.float32))
        return self.act_pdtype.pdfromflat(flat).sample().detach().numpy()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # 经验池够batch_size*max_episode_len=1024*25=25600
            return
        if t % 100 != 0:  # 每百训练步才更新一次
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        # NOTE 取完所有agent的o,a,r,o',done再单独取自己的
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        num_sample = 1
        target_q = 0.0
        # 贝尔曼方程的实现,done在加入经验池的时候由布尔转float了
        for no_use in range(num_sample):
            target_act_next_n = [agents[i].p_debug(obs_next_n[i], target_p_values=True).detach().numpy() for i in
                                 range(self.n)]
            target_q_next = self.q_debug(obs_next_n, target_act_next_n, target_q_values=True).detach().numpy()
            target_q += rew.reshape(target_q_next.shape) + self.args.gamma * target_q_next
        target_q /= num_sample

        q_loss = self.q_train(obs_n=obs_n, act_n=act_n, y=target_q, grad_norm_clipping=0.5)
        p_loss = self.p_train(obs_n=obs_n, act_n=act_n, grad_norm_clipping=0.5)

        self.make_update_exp(vals=self.p.parameters(), target_vals=self.target_p.parameters())
        self.make_update_exp(vals=self.q.parameters(), target_vals=self.target_q.parameters())
        # 返回值实际并没有被使用
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
