import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

    # 建立actor网络，构建训练的回调函数
def p_train(make_obs_ph_n, act_space_n,  # 第一项是所有agent的观察构成的状态空间，是一个个tensor，第二项所有agent的动作空间（list）
            p_index, p_func, q_func,  # 创建的actor 和 critic
            optimizer,  # 优化算法
            grad_norm_clipping=None,  # 梯度修剪
            local_q_func=False,   # 默认是MADDPG，当True时是DDPG
            num_units=64,  # 隐藏层神经元数量
            scope="trainer",  # tensorflow的集合范围，就是用的默认
            reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

        # create distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # 根据每个agent动作空间选择带size参数分布类型对象，

        # set up placeholders，构建分布的占位符，每个act_ph的形状是（None，pt.size）
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        p_input = obs_ph_n[p_index]

        # p是actor的网络输出，比如正态分布的μ和方差，p是个（act_dim行，act_pdtype.shape()列)的列表/数组，反正就是那个东西
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))  # 根据绝对命名路径获取范围内的变量

        # wrap parameters in distribution。先根据pd类型构建具体分布，再将网络输出的分布参数装载到agent对应的分布对象里，然后采样
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))  # 将分布的平方和，作为训练用的正则化项

        act_input_n = act_ph_n + []  # 创建一个新列表，而不是浅拷贝，拷贝动作占位符
        act_input_n[p_index] = act_pd.sample()  # *不是很理解为啥要对当前actor的动作进行采样？*:好像跟y的公式有关
        q_input = tf.concat(obs_ph_n + act_input_n, 1)  # 将每个agent的观察和动作合并成n个观察动作对
        if local_q_func:  # 如果是DDPG算法则q_input这么做
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)

        #计算Q值，构建损失值，计算梯度并裁剪
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]  # [:,0]裁取所有行的第0列，在这没意义
        pg_loss = -tf.reduce_mean(q)  # 训练actor的时候不需要更新Q网络，所以不用哪些均方误差，只要往Q值增大大方向就行
        loss = pg_loss + p_reg * 1e-3
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)  # 使用L2正则化对网络梯度裁剪
        # 返回一个优化器的操作，<tf.Operation 'agent_0_1/Adam' type=NoOp>

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())  # 构建一个(None,shape)的占位符

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
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

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):  # 计算基本损失与算法公式一致
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))  # 目标Q值也就是公式里的y

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
