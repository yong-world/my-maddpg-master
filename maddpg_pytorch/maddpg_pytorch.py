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
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False, device='cpu'):
        self.name = name
        self.n = len(obs_shape_n)  # 这里n是智能体个数，obs_shape_n=[34,34,34,34,28,28]
        self.agent_index = agent_index
        self.args = args
        self.device = device
        obs_ph_n = [torch.zeros(obs_shape_n[i]) for i in range(self.n)]

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

    def init_p_model_optimizer(self, act_space_n, obs_ph_n, agent_index, model, num_units, lr):
        act_pdtype = make_pdtype(act_space_n[agent_index])
        p_input = len(obs_ph_n[agent_index])  # p_index的索引是当前agent的编号

        # param_shape()是去输出动作的个数的和,因为输出是一个元素的列表所以需要[0]
        p = model(p_input, int(act_pdtype.param_shape()[0]), num_units=num_units)
        act_pd = act_pdtype.pdfromflat(p(torch.rand(p_input)))
        target_p = model(p_input, int(act_pdtype.param_shape()[0]), num_units=num_units)
        p_vars = p.parameters()
        p_optimizer = optim.Adam(p_vars, lr=lr)
        return p, p_optimizer, target_p, act_pdtype, act_pd

    def init_q_model_optimizer(self, act_space_n, obs_ph_n, lr, model, num_units):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # 获取输入的尺寸，包括每个agent的观察和各自的动作空间大小
        q_input = sum(i.numel() for i in obs_ph_n) + sum(i.numel() for i in act_ph_n)
        q = model(q_input, num_outputs=1, num_units=num_units)
        target_q = model(q_input, num_outputs=1, num_units=num_units)
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

        y = torch.tensor(y)
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
        act_output = self.p(obs_n[self.agent_index])
        # act_output_sample = torch.empty(act_output.shape)
        # for i in range(act_output.shape[0]):
        #     act_output_sample[i] = self.act_pdtype.pdfromflat(act_output[i]).sample()
        act_output_sample = self.act_pd.sample(act_output)
        act_n[self.agent_index] = act_output_sample

        batch_input = torch.cat([torch.cat(obs_n, dim=1), torch.cat(act_n, dim=1)], dim=1)
        q_loss = -torch.mean(self.q(batch_input))
        p_reg = torch.mean(torch.square(act_output))
        loss = q_loss + p_reg * 1e-3
        # print('ptrainqloss:{}\tp_reg:{}'.format(q_loss, p_reg))
        self.p_optimizer.zero_grad()
        loss.backward()
        if grad_norm_clipping is not None:
            for parameter in list(self.p.parameters()):
                torch.nn.utils.clip_grad_norm_(parameter, grad_norm_clipping)
        self.p_optimizer.step()
        return loss

    def action(self, obs):
        # 输入观察输出动作分布的采样
        flat = self.p(torch.from_numpy(obs).to(torch.float32))
        # return self.act_pdtype.pdfromflat(flat).sample().detach().numpy()
        return self.act_pd.sample(flat).detach().cpu().numpy()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def sample_from_replay(self, agents):
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)

        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(torch.from_numpy(obs))
            obs_next_n.append(torch.from_numpy(obs_next))
            act_n.append(torch.from_numpy(act))
        # NOTE 取完所有agent的o,a,r,o',done再单独取自己的
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        return obs_n, obs_next_n, act_n, obs, act, rew, obs_next, done

    def berman(self, agents, obs_next_n, rew):
        num_sample = 1
        target_q = 0.0
        # for no_use in range(num_sample):
        #     for i in range(self.n):
        #         flat=agents[i].p_debug(obs_next_n[i], target_p_values=True)
        #         target_act_next_n.append(torch.as_tensor([agents[i].act_pdtype.pdfromflat(flat[j]).sample().detach().numpy() for j in range(flat.shape[0])],dtype=np.float32,copy=False))
        # 贝尔曼方程的实现,done在加入经验池的时候由布尔转float了
        target_act_next_n = []
        # for no_use in range(num_sample):
        #     for i in range(self.n):
        #         flat = agents[i].p_debug(obs_next_n[i], target_p_values=True)
        #         target_act_next_sample = torch.empty((self.args.batch_size, agents[i].act_pdtype.param_shape()[0]))
        #         for j in range(self.args.batch_size):
        #             target_act_next_sample[j] = agents[i].act_pdtype.pdfromflat(flat[j]).sample().detach()
        #         target_act_next_n.append(target_act_next_sample)
        #     target_q_next = self.q_debug(obs_next_n, target_act_next_n, target_q_values=True).detach().numpy()
        #     target_q += rew.reshape(target_q_next.shape) + self.args.gamma * target_q_next
        # target_q /= num_sample
        for no_use in range(num_sample):
            for i in range(self.n):
                flat = agents[i].p_debug(obs_next_n[i], target_p_values=True)
                target_act_next_n.append(agents[i].act_pd.sample(flat))
            target_q_next = self.q_debug(obs_next_n, target_act_next_n, target_q_values=True).detach().numpy()
            target_q += rew.reshape(target_q_next.shape) + self.args.gamma * target_q_next
        target_q /= num_sample
        return target_q

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # 经验池够batch_size*max_episode_len=1024*25=25600
            return
        if t % 100 != 0:  # 每百训练步才更新一次
            return
        obs_n, obs_next_n, act_n, obs, act, rew, obs_next, done = self.sample_from_replay(agents)
        target_q = self.berman(agents, obs_next_n, rew)

        q_loss = self.q_train(obs_n=obs_n, act_n=act_n, y=target_q, grad_norm_clipping=0.5)
        p_loss = self.p_train(obs_n=obs_n, act_n=act_n, grad_norm_clipping=0.5)
        # print("q_loss: ", q_loss.data, "p_loss: ", p_loss.data)
        self.make_update_exp(vals=self.p.parameters(), target_vals=self.target_p.parameters())
        self.make_update_exp(vals=self.q.parameters(), target_vals=self.target_q.parameters())
        # 返回值实际并没有被使用
        return True
