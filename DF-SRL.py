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
    """
    Experience replay buffer for storing and sampling transitions in reinforcement learning.

    This buffer stores state transitions, rewards, actions, and terminal flags.
    It supports circular buffer functionality when capacity is reached.
    """
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        """
        Initialize the replay buffer with specified capacity and dimensions.

        Args:
            max_len (int): Maximum capacity of the buffer
            state_dim (int or tuple): Dimension of state space
            action_dim (int): Dimension of action space
            gpu_id (int): GPU device ID to use, -1 for CPU
        """
        self.now_len = 0  # Current number of transitions stored
        self.next_idx = 0  # Index for next insertion
        self.if_full = False  # Whether buffer has reached capacity
        self.max_len = max_len  # Maximum buffer capacity
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        # Buffer for rewards, terminal flags, and actions
        other_dim = 1 + 1 + self.action_dim  # reward + terminal + action dimensions
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        # Buffer for states, handling different state space types
        if isinstance(state_dim, int):  # For vector state spaces
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):  # For image-based state spaces
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim must be int or tuple')

    def extend_buffer(self, state, other):
        """
        Add multiple transitions to the buffer.

        Args:
            state (array): Batch of states to add
            other (array): Batch of [reward, terminal, action] data to add
        """
        size = len(other)
        next_idx = self.next_idx + size

        # Handle case where buffer wraps around
        if next_idx > self.max_len:
            # Fill until the end of buffer
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            # Wrap around and fill from beginning
            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            # Normal case: add to buffer sequentially
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample

        Returns:
            tuple: (rewards, terminal_flags, actions, states, next_states)
        """
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # rewards
                r_m_a[:, 1:2],  # terminal flags
                r_m_a[:, 2:],   # actions
                self.buf_state[indices],  # states
                self.buf_state[indices + 1])  # next states

    def update_now_len(self):
        """Update the current length of valid data in the buffer."""
        self.now_len = self.max_len if self.if_full else self.next_idx


class Arguments:
    """
    Configuration class for the DF-SRL algorithm.

    This class contains all hyperparameters and settings for training,
    evaluation, and saving/loading of the DF-SRL algorithm.
    """

    def __init__(self, agent=None, env=None):
        """
        Initialize the configuration with default values.

        Args:
            agent: The reinforcement learning agent
            env: The environment for training and evaluation
        """
        # Basic settings
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env      # The environment for training
        self.plot_shadow_on = False
        self.cwd = None     # Working directory for saving results
        self.if_remove = False  # Whether to remove existing results directory
        self.visible_gpu = '0,1,2,3'  # GPU devices to use
        self.worker_num = 4  # Number of parallel workers
        self.num_threads = 8  # Number of CPU threads

        # Training hyperparameters
        self.num_episode = 1000  # Number of episodes for training
        self.gamma = 0.995       # Discount factor for future rewards
        self.learning_rate = 1e-4  # Learning rate for optimizer
        self.soft_update_tau = 1e-2  # Rate for soft update of target networks

        # Network and buffer parameters
        self.net_dim = 256       # Neural network width
        self.batch_size = 512    # Batch size for training
        self.repeat_times = 2**2  # Number of updates per data collection
        self.target_step = 2000   # Steps to collect before updating network
        self.max_memo = 500000    # Capacity of replay buffer
        self.if_per_or_gae = False  # Whether to use Prioritized Experience Replay

        # Exploration parameters
        self.explorate_decay = 0.999  # Decay rate for exploration
        self.explorate_min = 0.2      # Minimum exploration rate

        # Evaluation parameters
        self.random_seed_list = [2234]  # Random seeds for reproducibility
        self.run_name = 'test'          # Name for this run (used in directory naming)

        # Saving and visualization settings
        self.train = True              # Whether to train the agent
        self.save_network = True       # Whether to save the trained network
        self.test_network = True       # Whether to test the network after training
        self.save_test_data = True     # Whether to save test results
        self.compare_with_pyomo = True  # Whether to compare with Pyomo optimization
        self.plot_on = True            # Whether to generate plots
        self.update_training_data = True  # Whether to update training data

    def init_before_training(self, if_main):
        """
        Initialize environment before training starts.

        This method sets up the working directory, random seeds,
        and GPU configuration before training begins.

        Args:
            if_main (bool): Whether this is the main process
        """
        # Set up working directory
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.run_name}'

        # Handle directory creation/removal
        if if_main:
            import shutil
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        # Configure GPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)


class AgentBase:
    """
    Base class for reinforcement learning agents.

    This class provides common functionality for RL agents, including
    action selection, environment exploration, and network updates.
    """
    def __init__(self):
        """Initialize the base agent with default values."""
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = None
        self.trajectory_list = None
        self.explore_rate = 1.0  # Initial exploration rate

        self.criterion = torch.nn.SmoothL1Loss()  # Loss function for value networks

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        """
        Initialize networks and optimizers.

        Args:
            net_dim (int): Width of neural networks
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            learning_rate (float): Learning rate for optimizers
            _if_per_or_gae (bool): Whether to use PER or GAE
            gpu_id (int): GPU device ID to use, -1 for CPU
        """
        # Set device (CPU or GPU)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        # Initialize critic and actor networks
        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri

        # Initialize target networks if needed
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        # Initialize optimizers
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri

        # Clean up temporary class references
        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        """
        Select an action based on the current state.

        Args:
            state (np.ndarray): Current state observation

        Returns:
            np.ndarray: Selected action
        """
        # Convert state to tensor and get action from policy network
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]

        # Add exploration noise if needed
        if rd.rand() < self.explore_rate:
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)

        return action.detach().cpu().numpy()

    def explore_env(self, env, target_step):
        """
        Explore the environment and collect transitions.

        Args:
            env: Environment to explore
            target_step (int): Number of steps to collect

        Returns:
            list: Collected trajectory of (state, (reward, done, action)) tuples
        """
        trajectory = list()

        # Reset environment to start new episode
        state = env.reset()

        # Collect transitions for target_step steps
        for _ in range(target_step):
            # Select action and get safe version from environment
            action = self.select_action(state)
            safe_action = env.get_safe_action(action)

            # Take step in environment
            state, next_state, reward, done = env.step(safe_action)

            # Store transition in trajectory
            trajectory.append((state, (reward, done, *safe_action)))

            # Reset environment if episode ended
            state = env.reset() if done else next_state

        return trajectory

    @staticmethod
    def optim_update(optimizer, objective):
        """
        Perform a single optimization step.

        Args:
            optimizer: PyTorch optimizer
            objective: Loss function to minimize
        """
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        Soft update target network parameters.

        Formula: θ_target = τ*θ_current + (1-τ)*θ_target

        Args:
            target_net: Target network to update
            current_net: Current network with new parameters
            tau (float): Update rate, typically small (e.g., 0.01)
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class CriticTwin(nn.Module):
    """
    Twin Critic network for TD3 algorithm.

    Implements two Q-networks to reduce overestimation bias in Q-learning.
    Uses shared feature extraction layers followed by separate Q-value heads.
    """
    def __init__(self, mid_dim, state_dim, action_dim):
        """
        Initialize the twin critic networks.

        Args:
            mid_dim (int): Hidden layer dimension
            state_dim (int): State space dimension
            action_dim (int): Action space dimension
        """
        super().__init__()
        # Shared feature extraction network for state-action pairs
        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU()
        )

        # First Q-network
        self.net_q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
            nn.Linear(mid_dim, 1)
        )

        # Second Q-network
        self.net_q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
            nn.Linear(mid_dim, 1)
        )

    def forward(self, state, action):
        """
        Forward pass through the first Q-network.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            torch.Tensor: Q-value from the first network
        """
        features = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(features)

    def get_q1_q2(self, state, action):
        """
        Get Q-values from both networks.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            tuple: (Q1-value, Q2-value)
        """
        features = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(features), self.net_q2(features)


class Actor(nn.Module):
    """
    Actor network for TD3 algorithm.

    Maps states to deterministic actions using a neural network.
    Includes methods for both deterministic and noisy action selection.
    """
    def __init__(self, mid_dim, state_dim, action_dim):
        """
        Initialize the actor network.

        Args:
            mid_dim (int): Hidden layer dimension
            state_dim (int): State space dimension
            action_dim (int): Action space dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
            nn.Linear(mid_dim, action_dim)
        )

    def forward(self, state):
        """
        Forward pass to get deterministic actions.

        Args:
            state: State tensor

        Returns:
            torch.Tensor: Action values in range [-1, 1]
        """
        return self.net(state).tanh()

    def get_action(self, state, action_std):
        """
        Get actions with exploration noise.

        Args:
            state: State tensor
            action_std (float): Standard deviation of noise

        Returns:
            torch.Tensor: Noisy actions clamped to [-1, 1]
        """
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class AgentTD3(AgentBase):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

    Implements the TD3 algorithm with:
    - Twin critics to reduce overestimation bias
    - Delayed policy updates
    - Target policy smoothing
    """
    def __init__(self):
        """Initialize the TD3 agent with specific parameters."""
        super().__init__()
        self.explore_noise = 0.3  # Standard deviation for exploration
        self.policy_noise = 0.1   # Standard deviation for target policy smoothing
        self.update_freq = 2      # Policy update frequency (delayed updates)
        self.if_use_cri_target = self.if_use_act_target = True  # Use target networks
        self.ClassCri = CriticTwin  # Critic class to use
        self.ClassAct = Actor      # Actor class to use

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """
        Update critic and actor networks.

        Args:
            buffer: Replay buffer containing experiences
            batch_size (int): Batch size for training
            repeat_times (int): Number of updates per call
            soft_update_tau (float): Rate for soft target updates

        Returns:
            tuple: (critic_loss, actor_loss)
        """
        buffer.update_now_len()
        obj_critic = obj_actor = None

        # Perform multiple updates based on buffer size and repeat_times
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            # Update critic networks
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            # Delayed policy updates (less frequent than critic updates)
            if update_c % self.update_freq == 0:
                # Calculate actor loss using critic target network
                action_pg = self.act(state)
                obj_actor = -self.cri_target(state, action_pg).mean()
                self.optim_update(self.act_optim, obj_actor)

                # Update target networks
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        """
        Calculate critic loss.

        Args:
            buffer: Replay buffer
            batch_size (int): Batch size

        Returns:
            tuple: (critic_loss, states)
        """
        with torch.no_grad():
            # Sample batch from buffer
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            # Get next actions with noise (target policy smoothing)
            next_a = self.act_target.get_action(next_s, self.policy_noise)

            # Calculate target Q-value using minimum of twin Q-values
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * next_q

        # Calculate critic loss for both Q-networks
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state

    def _update_exploration_rate(self, explorate_decay, explore_rate_min):
        """
        Update exploration rate with decay.

        Args:
            explorate_decay (float): Rate at which exploration decays
            explore_rate_min (float): Minimum exploration rate
        """
        self.explore_rate = max(self.explore_rate * explorate_decay, explore_rate_min)


def update_buffer(_trajectory):
    """
    Process trajectory data and add it to the replay buffer.

    This function converts trajectory data (list of state-reward-action tuples)
    into tensor format and adds it to the replay buffer for training.

    Args:
        _trajectory (list): List of (state, (reward, done, action)) tuples

    Returns:
        tuple: (steps_count, mean_reward)
    """
    # Convert states to tensor
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)

    # Convert rewards, done flags, and actions to tensor
    ary_other = torch.as_tensor([item[1] for item in _trajectory])

    # Add data to replay buffer
    buffer.extend_buffer(ten_state, ary_other)

    # Calculate statistics
    _steps = ten_state.shape[0]  # Number of steps in trajectory
    _r_exp = ary_other[:, 0].mean()  # Average reward

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
