import copy as cp
import os
import pickle
from copy import deepcopy

import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn
import wandb


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        '''get reward, mask, action, state, next_state,
        actually, next_state is calculated based on state_indice,
        we need to randomly choose more blocks, instead of justing random choose state'''
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx


class Arguments:
    '''revise here for our own purpose'''

    def __init__(self, agent=None, env=None):

        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training
        self.plot_shadow_on = False
        self.cwd = None
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        # self.replace_train_data=True
        self.visible_gpu = '0,1,2,3'
        self.worker_num = 4
        self.num_threads = 8

        '''Arguments for training'''
        self.num_episode = 1000
        self.gamma = 0.995  # discount factor of future rewards
        self.learning_rate = 1e-4  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 1e-2  # 2 ** -8 ~= 5e-3

        self.net_dim = 256  # the network width 256
        self.batch_size = 512  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 2  # repeatedly update network to keep critic's loss small
        self.target_step = 2000  # collect target_step experiences , then update network, 1024
        self.max_memo = 500000  # capacity of replay buffer
        self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.
        ## arguments for controlling exploration
        self.explorate_decay = 0.999
        self.explorate_min = 0.2
        '''Arguments for evaluate'''
        self.random_seed_list = [2234]
        self.run_name = 'test'

        '''Arguments for save and plot issues'''
        self.train = True
        self.save_network = True
        self.test_network = True
        self.save_test_data = True
        self.compare_with_pyomo = True
        self.plot_on = True
        self.update_training_data = True

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.run_name}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)  # control how many GPU is used


class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = None
        self.trajectory_list = None
        self.explore_rate = 1.0

        self.criterion = torch.nn.SmoothL1Loss()

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        # explict call self.init() for multiprocessing
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        if rd.rand() < self.explore_rate:
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu().numpy()

    def explore_env(self, env, target_step):
        '''return (state,(reward,done,*action))'''
        trajectory = list()
        # state_list = []
        # next_state_list = []

        state = env.reset()
        # state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            safe_action = env.get_safe_action(action)
            state, next_state, reward, done, = env.step(safe_action)
            # print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time},reward is:{reward}')
            trajectory.append((state, (reward, done, *safe_action)))
            state = env.reset() if done else next_state

        # self.state = state
        return trajectory

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class AgentTD3(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.3  # standard deviation of exploration noise
        self.policy_noise = 0.1  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = CriticTwin
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            if update_c % self.update_freq == 0:  # delay update
                # delayed policy update
                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.cri_target(state,
                                             action_pg).mean()  # use cri_target instead of cri for stable training
                self.optim_update(self.act_optim, obj_actor)

                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

    def _update_exploration_rate(self, explorate_decay, explore_rate_min):
        self.explore_rate = max(self.explore_rate * explorate_decay, explore_rate_min)
        '''this function is used to update the explorate probability when select action'''


def update_buffer(_trajectory):
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
    ary_other = torch.as_tensor([item[1] for item in _trajectory])
    ary_other[:, 0] = ary_other[:, 0]  # ten_reward
    # ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma
    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
    return _steps, _r_exp





from safe_battery_env import PowerNetEnv

if __name__ == '__main__':
    args = Arguments()
    reward_record = {'episode': [], 'steps': [], 'mean_episode_reward': [], 'violation_time': [],
                     'violation_value': []}
    loss_record = {'episode': [], 'steps': [], 'critic_loss': [], 'actor_loss': [], 'entropy_loss': []}
    args.visible_gpu = '0'
    for seed in args.random_seed_list:
        args.random_seed = seed
        # set different seed
        args.agent = AgentTD3()
        agent_name = f'{args.agent.__class__.__name__}'
        args.agent.cri_target = True
        args.init_before_training(if_main=True)
        args.env = PowerNetEnv()
        agent = args.agent
        env = args.env
        agent.init(args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate,
                   args.if_per_or_gae)
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_space.shape[0],
                              action_dim=env.action_space.shape[0])
        '''start training'''
        cwd = args.cwd
        gamma = args.gamma
        batch_size = args.batch_size  # how much data should be used to update net
        target_step = args.target_step  # how manysteps of one episode should stop
        repeat_times = args.repeat_times  # how many times should update for one batch size data
        soft_update_tau = args.soft_update_tau

        env.init()
        agent.state = env.reset()
        '''collect data and train and update network'''
        num_episode = args.num_episode
        wandb.init(project='DF_SRL_ENV', name=args.run_name, settings=wandb.Settings(start_method="fork"))
        wandb.config = {
            "epochs": num_episode,
            "batch_size": batch_size,
            'parameter_good_action': 200,
            'parameter_penalty': 1000}
        wandb.define_metric('custom_step')
        if args.train:
            collect_data = True
            while collect_data:
                print(f'buffer:{buffer.now_len}')
                with torch.no_grad():
                    trajectory = agent.explore_env(env, target_step)
                    steps, r_exp = update_buffer(trajectory)
                    buffer.update_now_len()
                if buffer.now_len >= 10000:
                    collect_data = False
            for i_episode in range(num_episode):
                critic_loss, actor_loss = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
                ## add wandb record

                wandb.log({'critic loss': critic_loss, 'custom_step': i_episode})
                wandb.log({'actor loss': actor_loss, 'custom_step': i_episode})

                loss_record['critic_loss'].append(critic_loss)
                loss_record['actor_loss'].append(actor_loss)

    act_save_path = f'{args.cwd}/actor.pth'
    cri_save_path = f'{args.cwd}/critic.pth'
    print('training data have been saved')
    if args.save_network:
        torch.save(agent.act.state_dict(), act_save_path)
        torch.save(agent.cri.state_dict(), cri_save_path)
        print('actor and critic parameters have been saved')
    print('training finished')
    wandb.finish()
