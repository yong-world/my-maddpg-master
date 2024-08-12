import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from maddpg import AgentTrainer
from maddpg_pytorch.replay_buffer import ReplayBuffer
from maddpg_pytorch.distributions_pytorch import make_pdtype
# from maddpg.common.distributions import make_pdtype
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    for var, var_target in zip(vals, target_vals):
        var_target.data.copy_(polyak * var_target.data + (1.0 - polyak) * var.data)

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func,grad_norm_clipping=None, local_q_func=False, num_units=64, lr=1e-2):
    # 功能：
    # 返回：动作分布的采样函数，网络的训练函数，actor网络软更新函数，(当前actor网络输出值函数,目标actor网络输出值函数)debug数据元组
    # ###
    act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
    obs_ph_n = make_obs_ph_n
    act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
    p_input = obs_ph_n[p_index]

    p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), num_units=num_units)
    p_func_vars = list(p_func.parameters())

    act_pd = act_pdtype_n[p_index].pdfromflat(p)
    act_sample = act_pd.sample()
    p_reg = torch.mean(torch.square(act_pd.flatparam()))

    act_input_n = act_ph_n[:]
    act_input_n[p_index] = act_pd.sample()
    q_input = torch.cat(obs_ph_n + act_input_n, dim=1)
    if local_q_func:
        q_input = torch.cat([obs_ph_n[p_index], act_input_n[p_index]], dim=1)

    q = q_func(q_input, 1, reuse=True, num_units=num_units)[:, 0]
    pg_loss = -torch.mean(q)
    loss = pg_loss + p_reg * 1e-3
    optimizer = optim.Adam(p_func_vars, lr=lr)
    optimizer.zero_grad()
    loss.backward()
    if grad_norm_clipping is not None:
        torch.nn.utils.clip_grad_norm_(p_func_vars, grad_norm_clipping)
    optimizer.step()

    def train_fn(*inputs):
        optimizer.zero_grad()
        loss_val = loss(*inputs)
        loss_val.backward()
        if grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(p_func_vars, grad_norm_clipping)
        optimizer.step()
        return loss_val

    def act_fn(obs):
        with torch.no_grad():
            return act_sample(obs).numpy()

    def p_values_fn(obs):
        with torch.no_grad():
            return p(obs).numpy()

    target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), num_units=num_units)
    target_p_func_vars = list(target_p.parameters())
    update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

    def target_act_fn(obs):
        with torch.no_grad():
            return act_pdtype_n[p_index].pdfromflat(target_p).sample().numpy()

    return act_fn, train_fn, update_target_p, {'p_values': p_values_fn, 'target_act': target_act_fn}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func,grad_norm_clipping=None, local_q_func=False,num_units=64,lr=1e-2):
    act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
    obs_ph_n = make_obs_ph_n
    act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
    target_ph = torch.tensor([], dtype=torch.float32)

    q_input = torch.cat(obs_ph_n + act_ph_n, dim=1)
    if local_q_func:
        q_input = torch.cat([obs_ph_n[q_index], act_ph_n[q_index]], dim=1)
    q = q_func(q_input, 1, num_units=num_units)[:, 0]
    q_func_vars = list(q_func.parameters())

    q_loss = torch.mean(torch.square(q - target_ph))

    q_reg = torch.mean(torch.square(q))
    loss = q_loss  # + 1e-3 * q_reg
    optimizer = optim.Adam(q_func_vars, lr=lr)
    optimizer.zero_grad()
    loss.backward()
    if grad_norm_clipping is not None:
        torch.nn.utils.clip_grad_norm_(q_func_vars, grad_norm_clipping)
    optimizer.step()

    def train_fn(*inputs):
        optimizer.zero_grad()
        loss_val = loss(*inputs)
        loss_val.backward()
        if grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(q_func_vars, grad_norm_clipping)
        optimizer.step()
        return loss_val

    def q_values_fn(*inputs):
        with torch.no_grad():
            return q(*inputs).numpy()

    target_q = q_func(q_input, 1, num_units=num_units)[:, 0]
    target_q_func_vars = list(target_q.parameters())
    update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

    def target_q_values_fn(*inputs):
        with torch.no_grad():
            return target_q(*inputs).numpy()

    return train_fn, update_target_q, {'q_values': q_values_fn, 'target_q_values': target_q_values_fn}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = [torch.zeros(obs_shape_n[i]) for i in range(self.n)]

        self.q_train, self.q_update, self.q_debug = q_train(
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            lr=args.lr
        )

        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            lr=args.lr
        )

        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return
        if t % 100 != 0:  # 每百步骤更新一次
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
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        num_sample = 1
        target_q = 0.0
        # TODO p_debug和q_debug函数，p_train，q_train，p、qupdata，
        for _ in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()
        # 返回值实际并没有被使用
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
