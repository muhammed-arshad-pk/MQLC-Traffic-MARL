import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import networkx as nx
from torch.nn.functional import relu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import gc
import sys
try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not installed. Memory monitoring disabled.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class TrafficEnv:
    def __init__(self, num_avs, num_hdvs, num_lanes=3, road_length=100.0, condition='sparse'):
        try:
            self.num_avs = num_avs
            self.num_hdvs = num_hdvs
            self.num_vehicles = num_avs + num_hdvs
            self.num_lanes = num_lanes
            self.road_length = road_length
            self.lane_width = 4.0
            self.vehicle_size = 2.0
            self.action_space = ['lane_left', 'idle', 'lane_right', 'faster', 'slower']
            self.state_dim = self.num_vehicles * 5
            self.observation_dim = 7
            self.max_speed = 32.0
            self.min_speed = 20.0
            self.collision_distance = 1.0 if condition == 'sparse' else 1.2
            self.condition = condition
            if condition == 'sparse':
                self.collision_threshold = max(25, int(15 + 0.5 * self.num_vehicles))
            elif condition == 'normal':
                self.collision_threshold = max(18, int(10 + 0.8 * self.num_vehicles))
            else:  # dense
                self.collision_threshold = max(15, int(8 + 1.0 * self.num_vehicles))
            self.collisions = 0
            self.total_steps = 0
            self.episode_lengths = []
            self.avg_speeds = []
            self.episode_rewards = []
            self.episode_collisions_list = []
            logger.info(f"TrafficEnv initialized: {num_avs} AVs, {num_hdvs} HDVs, condition: {condition}, collision threshold: {self.collision_threshold}")
        except Exception as e:
            logger.error(f"Error in TrafficEnv init: {e}")
            raise e
    
    def reset(self):
        try:
            self._state = np.zeros(self.state_dim)
            observations = np.zeros((self.num_vehicles, self.observation_dim))
            x_positions = np.linspace(0, self.road_length * 1.5, self.num_vehicles + 1)[:-1]
            np.random.shuffle(x_positions)
            for i in range(self.num_vehicles):
                self._state[i*5] = 1.0
                self._state[i*5 + 1] = x_positions[i] % self.road_length
                self._state[i*5 + 2] = np.random.randint(0, self.num_lanes) * self.lane_width
                self._state[i*5 + 3] = np.random.uniform(25.0, self.max_speed)
                self._state[i*5 + 4] = 0.0
                observations[i, :5] = self._state[i*5:i*5+5]
                observations[i, 5:7] = np.random.rand(2)
            self.episode_collisions = 0
            self.episode_steps = 0
            self.episode_reward = 0
            logger.info("Environment reset")
            return self._state, observations
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            raise e
    
    def hdv_policy(self, state, vehicle_idx):
        try:
            x = state[vehicle_idx*5 + 1]
            y = state[vehicle_idx*5 + 2]
            vx = state[vehicle_idx*5 + 3]
            lane = y / self.lane_width
            
            min_dist = float('inf')
            for j in range(self.num_vehicles):
                if j == vehicle_idx:
                    continue
                x_j = state[j*5 + 1]
                y_j = state[j*5 + 2]
                dist = np.sqrt(((x - x_j) % self.road_length)**2 + (y - y_j)**2)
                if dist < min_dist:
                    min_dist = dist
            
            density_factor = self.num_vehicles / (self.road_length * self.num_lanes * self.lane_width)
            slow_threshold = 5.0 + 1.5 * density_factor if self.condition == 'sparse' else 8.0 + 4.0 * density_factor
            lane_change_prob = 0.002 if self.condition in ['normal', 'dense'] else 0.005
            speed_prob = 0.1 if self.condition == 'sparse' else 0.01
            
            if min_dist < slow_threshold:
                action = 'slower'
            elif random.random() < lane_change_prob:
                if random.random() < 0.5 and lane > 0:
                    action = 'lane_left'
                elif lane < self.num_lanes - 1:
                    action = 'lane_right'
                else:
                    action = 'idle'
            elif random.random() < speed_prob:
                action = 'faster'
            else:
                action = 'idle'
            return self.action_space.index(action)
        except Exception as e:
            logger.error(f"Error in hdv_policy: {e}")
            raise e
    
    def step(self, av_actions):
        try:
            self.episode_steps += 1
            self.total_steps += 1
            state = np.zeros(self.state_dim)
            observations = np.zeros((self.num_vehicles, self.observation_dim))
            rewards = [0.0] * self.num_avs
            collisions = 0
            
            prev_state = self._state.copy()
            
            density_factor = self.num_vehicles / (self.road_length * self.num_lanes * self.lane_width)
            w1, w2 = 0.65, 0.05
            w3 = 0.3 if self.condition == 'sparse' else 0.2 if self.condition == 'normal' else 0.1
            w4 = 0.1 if self.condition == 'sparse' else 0.2
            
            hdv_actions = [self.hdv_policy(prev_state, i) for i in range(self.num_avs, self.num_vehicles)]
            joint_action = av_actions + hdv_actions
            
            for i in range(self.num_vehicles):
                action = joint_action[i]
                presence = prev_state[i*5]
                x = prev_state[i*5 + 1]
                y = prev_state[i*5 + 2]
                vx = prev_state[i*5 + 3]
                vy = prev_state[i*5 + 4]
                
                if presence == 0:
                    continue
                
                x += vx * 0.5
                x %= self.road_length
                
                lane = y / self.lane_width
                if action == self.action_space.index('lane_left') and lane > 0:
                    y -= self.lane_width
                elif action == self.action_space.index('lane_right') and lane < self.num_lanes - 1:
                    y += self.lane_width
                elif action == self.action_space.index('faster'):
                    vx = min(vx + 3.0, self.max_speed)
                elif action == self.action_space.index('slower'):
                    vx = max(vx - 1.0, self.min_speed)
                
                state[i*5] = presence
                state[i*5 + 1] = x
                state[i*5 + 2] = y
                state[i*5 + 3] = vx
                state[i*5 + 4] = vy
                observations[i, :5] = state[i*5:i*5+5]
                observations[i, 5:7] = np.random.rand(2)
                
                if i < self.num_avs:
                    r1 = 50.0
                    r2 = -1.0 if joint_action[i] in [self.action_space.index('lane_left'), self.action_space.index('lane_right')] else 0.0
                    r3 = max(0.67, min(1.0, vx / self.max_speed)) * (4.0 if self.condition == 'sparse' else 2.5 if self.condition == 'normal' else 1.0)
                    avg_speed = np.mean(state[3::5])
                    r4 = 4.0 * (1 - min(abs(vx - avg_speed) / 10.0, 1.0))
                    rewards[i] = w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4
            
            x_positions = state[1::5].reshape(-1, 1)
            y_positions = state[2::5].reshape(-1, 1)
            x_dists = x_positions - x_positions.T
            x_dists = np.minimum(x_dists, self.road_length - x_dists)
            y_dists = y_positions - y_positions.T
            dists = np.sqrt(x_dists**2 + y_dists**2)
            dists = np.triu(dists, k=1)
            collision_mask = (dists < self.collision_distance) & (dists > 0)
            proximity_mask = (dists >= self.collision_distance) & (dists < (7.0 * self.collision_distance if self.condition == 'dense' else 6.0 * self.collision_distance))
            collisions = np.sum(collision_mask)
            
            collision_penalty = -20.0 if self.condition == 'sparse' else -15.0
            if collisions > 0:
                collision_pairs = np.where(collision_mask)
                for i, j in zip(collision_pairs[0], collision_pairs[1]):
                    if i < self.num_avs:
                        rewards[i] = w1 * collision_penalty + w2 * (rewards[i] - w1 * collision_penalty) / w1 + w3 * (rewards[i] - w1 * collision_penalty - w2 * (-1.0 if joint_action[i] in [self.action_space.index('lane_left'), self.action_space.index('lane_right')] else 0.0)) / w1
                    if j < self.num_avs:
                        rewards[j] = w1 * collision_penalty + w2 * (rewards[j] - w1 * collision_penalty) / w1 + w3 * (rewards[j] - w1 * collision_penalty - w2 * (-1.0 if joint_action[j] in [self.action_space.index('lane_left'), self.action_space.index('lane_right')] else 0.0)) / w1
            
            if np.sum(proximity_mask) > 0:
                proximity_pairs = np.where(proximity_mask)
                for i, j in zip(proximity_pairs[0], proximity_pairs[1]):
                    if i < self.num_avs:
                        rewards[i] -= 2.0
                    if j < self.num_avs:
                        rewards[j] -= 2.0
            
            self.collisions += collisions
            self.episode_collisions += collisions
            global_reward = sum(rewards)
            self.episode_reward += global_reward
            
            done = self.episode_steps >= 60 or self.episode_collisions > self.collision_threshold
            
            if done:
                self.episode_lengths.append(self.episode_steps)
                self.avg_speeds.append(np.mean(state[3::5]))
                self.episode_rewards.append(self.episode_reward / self.episode_steps)
                self.episode_collisions_list.append(self.episode_collisions)
            
            self._state = state
            if self.episode_steps % 20 == 0:
                logger.info(f"Step {self.episode_steps}, Reward={global_reward:.2f}, Collisions={collisions}")
            return state, observations, rewards, global_reward, done
        except Exception as e:
            logger.error(f"Error in step: {e}")
            raise e
    
    def get_traffic_conditions(self, state):
        try:
            m_velocity = np.mean(state[3::5])
            t_density = np.sum(state[0::5]) / self.num_vehicles
            s_variance = np.var(state[3::5])
            return m_velocity, t_density, s_variance
        except Exception as e:
            logger.error(f"Error in get_traffic_conditions: {e}")
            raise e

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        try:
            self.linear = nn.Linear(in_features, out_features)
            logger.info("GCNLayer initialized")
        except Exception as e:
            logger.error(f"Error in GCNLayer init: {e}")
            raise e
    
    def forward(self, x, adj):
        try:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            if len(adj.shape) == 2:
                adj = adj.unsqueeze(0)
            x = torch.bmm(adj, x)
            x = self.linear(x)
            return relu(x)
        except Exception as e:
            logger.error(f"Error in GCNLayer forward: {e}")
            raise e

class IndividualQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, num_vehicles, hidden_dim=32):
        super(IndividualQNetwork, self).__init__()
        try:
            self.num_vehicles = num_vehicles
            self.hidden_dim = hidden_dim
            self.gcn1 = GCNLayer(obs_dim, hidden_dim)
            self.mlp = nn.Linear(3, hidden_dim)
            self.fc = nn.Linear(hidden_dim + hidden_dim + 5, hidden_dim)
            self.output = nn.Linear(hidden_dim, action_dim * num_vehicles)
            logger.info("IndividualQNetwork initialized")
        except Exception as e:
            logger.error(f"Error in IndividualQNetwork init: {e}")
            raise e
    
    def forward(self, obs, adj, env_features):
        try:
            if len(obs.shape) == 2:
                obs = obs.unsqueeze(0)
            if len(env_features.shape) == 1:
                env_features = env_features.unsqueeze(0)
            h_sur = self.gcn1(obs, adj)
            h_env = self.mlp(env_features)
            h_ori = obs[:, :, :5] if len(obs.shape) == 3 else obs[:, :5]
            h_combined = torch.cat([h_sur.mean(dim=1), h_env, h_ori.mean(dim=1)], dim=-1)
            h = relu(self.fc(h_combined))
            return self.output(h).view(-1, self.num_vehicles, self.output.out_features // self.num_vehicles)
        except Exception as e:
            logger.error(f"Error in IndividualQNetwork forward: {e}")
            raise e

class GlobalQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_vehicles, hidden_dim=32):
        super(GlobalQNetwork, self).__init__()
        try:
            self.num_vehicles = num_vehicles
            self.hidden_dim = hidden_dim
            self.gcn1 = GCNLayer(state_dim // num_vehicles, hidden_dim)
            self.mlp = nn.Linear(3, hidden_dim)
            self.fc = nn.Linear(hidden_dim + hidden_dim + 5, hidden_dim)
            self.output = nn.Linear(hidden_dim, action_dim ** num_vehicles)
            logger.info("GlobalQNetwork initialized")
        except Exception as e:
            logger.error(f"Error in GlobalQNetwork init: {e}")
            raise e
    
    def forward(self, state, adj, env_features):
        try:
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(env_features.shape) == 1:
                env_features = env_features.unsqueeze(0)
            state = state.view(-1, self.num_vehicles, state.shape[-1] // self.num_vehicles)
            h_sur = self.gcn1(state, adj)
            h_env = self.mlp(env_features)
            h_ori = state[:, :, :5] if len(state.shape) == 3 else state[:, :5]
            h_combined = torch.cat([h_sur.mean(dim=1), h_env, h_ori.mean(dim=1)], dim=-1)
            h = relu(self.fc(h_combined))
            return self.output(h)
        except Exception as e:
            logger.error(f"Error in GlobalQNetwork forward: {e}")
            raise e

class MQLCAgent:
    def __init__(self, obs_dim, state_dim, action_dim, num_avs, num_vehicles, condition='sparse', **kwargs):
        try:
            self.action_dim = action_dim
            self.num_avs = num_avs
            self.num_vehicles = num_vehicles
            self.condition = condition
            self.gamma = kwargs.get('gamma', 0.995)
            self.lambda_reg = kwargs.get('lambda_reg', 0.05)
            self.epsilon = kwargs.get('epsilon', 1.0)
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.999 if condition == 'dense' else 0.997
            self.alpha = kwargs.get('alpha', 1.0)
            self.hidden_dim = kwargs.get('hidden_dim', 64 if condition == 'dense' else 32)
            self.lr = kwargs.get('lr', 0.0003 if condition == 'dense' else 0.0005)
            self.individual_q = IndividualQNetwork(obs_dim, action_dim, num_avs, self.hidden_dim).to(device)
            self.global_q = GlobalQNetwork(state_dim, action_dim, num_avs, self.hidden_dim).to(device)
            self.target_individual_q = IndividualQNetwork(obs_dim, action_dim, num_avs, self.hidden_dim).to(device)
            self.target_global_q = GlobalQNetwork(state_dim, action_dim, num_avs, self.hidden_dim).to(device)
            self.target_individual_q.load_state_dict(self.individual_q.state_dict())
            self.target_global_q.load_state_dict(self.global_q.state_dict())
            self.optimizer = optim.Adam(
                list(self.individual_q.parameters()) + list(self.global_q.parameters()), lr=self.lr
            )
            self.ind_buffer = deque(maxlen=1000)
            self.glo_buffer = deque(maxlen=1000)
            self.rewards = []
            self.ind_losses = []
            self.glo_losses = []
            self.reg_losses = []
            self.best_reward = -float('inf')
            logger.info("MQLCAgent initialized")
        except Exception as e:
            logger.error(f"Error in MQLCAgent init: {e}")
            raise e
    
    def compute_priority(self, m_velocity, t_density, s_variance):
        try:
            urgency = m_velocity + t_density + self.alpha * s_variance
            return 'high' if urgency > self.epsilon else 'low'
        except Exception as e:
            logger.error(f"Error in compute_priority: {e}")
            raise e
    
    def select_action(self, obs, state, adj, env_features, priorities):
        try:
            with torch.no_grad():
                if random.random() < self.epsilon:
                    actions = [random.randint(0, self.action_dim - 1) for _ in range(self.num_avs)]
                else:
                    obs = obs[:self.num_avs].to(device)
                    if len(obs.shape) == 2:
                        obs = obs.unsqueeze(0)
                    q_values = self.individual_q(obs, adj.to(device), env_features.to(device))
                    actions = []
                    for i, priority in enumerate(priorities[:self.num_avs]):
                        q_vals = q_values[0, i]
                        actions.append(q_vals.argmax().item())
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                return actions
        except Exception as e:
            logger.error(f"Error in select_action: {e}")
            raise e
    
    def select_joint_action(self, state, adj, env_features, actions_list):
        try:
            with torch.no_grad():
                q_values = self.global_q(state.to(device), adj.to(device), env_features.to(device))
                joint_action_idx = q_values.argmax().item()
                joint_action = []
                temp_idx = joint_action_idx
                for _ in range(self.num_avs):
                    joint_action.append(temp_idx % self.action_dim)
                    temp_idx //= self.action_dim
                return joint_action
        except Exception as e:
            logger.error(f"Error in select_joint_action: {e}")
            raise e
    
    def store_transition(self, obs, action, reward, next_obs, state, joint_action, global_reward, next_state, vehicle_idx):
        try:
            self.ind_buffer.append((
                torch.tensor(obs[:self.num_avs], dtype=torch.float32).to(device),
                torch.tensor(action, dtype=torch.long).to(device),
                torch.tensor(reward, dtype=torch.float32).to(device),
                torch.tensor(next_obs[:self.num_avs], dtype=torch.float32).to(device),
                vehicle_idx
            ))
            self.glo_buffer.append((
                torch.tensor(state, dtype=torch.float32).to(device),
                torch.tensor(joint_action, dtype=torch.long).to(device),
                torch.tensor(global_reward, dtype=torch.float32).to(device),
                torch.tensor(next_state, dtype=torch.float32).to(device)
            ))
        except Exception as e:
            logger.error(f"Error in store_transition: {e}")
            raise e
    
    def save_model(self, episode, avg_reward, condition):
        try:
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                torch.save(self.individual_q.state_dict(), f'ind_q_{condition}_ep{episode}.pth')
                torch.save(self.global_q.state_dict(), f'glo_q_{condition}_ep{episode}.pth')
                logger.info(f"Saved model at episode {episode} for {condition} with avg reward {avg_reward:.2f}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e
    
    def update(self, batch_size=32):
        try:
            if len(self.ind_buffer) < batch_size or len(self.glo_buffer) < batch_size:
                return
            
            ind_batch = random.sample(self.ind_buffer, batch_size)
            obs, actions, rewards, next_obs, vehicle_indices = zip(*ind_batch)
            obs = torch.stack(obs).to(device)
            actions = torch.stack(actions).to(device)
            rewards = torch.stack(rewards).to(device)
            next_obs = torch.stack(next_obs).to(device)
            vehicle_indices = torch.tensor(vehicle_indices, dtype=torch.long).to(device)
            adj = torch.ones((self.num_avs, self.num_avs)).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
            env_features = torch.tensor([[1.0, 0.5, 0.2] for _ in range(batch_size)], dtype=torch.float32).to(device)
            
            q_values = self.individual_q(obs, adj, env_features)
            q_values = q_values[torch.arange(batch_size), vehicle_indices]
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1).detach()
            
            next_q_values = self.target_individual_q(next_obs, adj, env_features)
            next_q_values = next_q_values[torch.arange(batch_size), vehicle_indices]
            next_q_values = next_q_values.max(1)[0].detach()
            targets = rewards + self.gamma * next_q_values
            ind_loss = ((q_values - targets) ** 2).mean()
            
            glo_batch = random.sample(self.glo_buffer, batch_size)
            states, joint_actions, global_rewards, next_states = zip(*glo_batch)
            states = torch.stack(states).to(device)
            joint_actions = torch.stack(joint_actions).to(device)
            global_rewards = torch.stack(global_rewards).to(device)
            next_states = torch.stack(next_states).to(device)
            
            joint_action_indices = torch.zeros(batch_size, dtype=torch.long).to(device)
            for i in range(self.num_avs):
                joint_action_indices += joint_actions[:, i] * (self.action_dim ** i)
            
            q_values_glo = self.global_q(states, adj, env_features)
            q_values_glo = q_values_glo.gather(1, joint_action_indices.unsqueeze(1)).squeeze(1).detach()
            next_q_values_glo = self.target_global_q(next_states, adj, env_features).max(1)[0].detach()
            targets_glo = global_rewards + self.gamma * next_q_values_glo
            glo_loss = ((q_values_glo - targets_glo) ** 2).mean()
            
            ind_q_values = self.individual_q(obs, adj, env_features)
            ind_q_sum = torch.zeros(batch_size, dtype=torch.float32).to(device)
            for i in range(self.num_avs):
                vehicle_idx = torch.full((batch_size,), i, dtype=torch.long).to(device)
                action_idx = joint_actions[:, i]
                q_val = ind_q_values[torch.arange(batch_size), vehicle_idx, action_idx]
                ind_q_sum += q_val
            reg_loss = ((q_values_glo - ind_q_sum) ** 2).mean()
            
            self.ind_losses.append(ind_loss.item())
            self.glo_losses.append(glo_loss.item())
            self.reg_losses.append(reg_loss.item())
            
            total_loss = glo_loss + ind_loss + self.lambda_reg * reg_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            if len(self.ind_losses) % 100 == 0:
                logger.info(f"Update: ind_loss={ind_loss.item():.4f}, glo_loss={glo_loss.item():.4f}, reg_loss={reg_loss.item():.4f}")
        except Exception as e:
            logger.error(f"Error in update: {e}")
            raise e

def train_mqlc_for_condition(num_avs, num_hdvs, condition, episodes=100, batch_size=32):
    try:
        log_file = f'train_mqlc_{condition}.log'
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        logger.info(f"Starting training for {condition} condition: {num_avs} AVs, {num_hdvs} HDVs")
        env = TrafficEnv(num_avs, num_hdvs, condition=condition)
        episodes = 100  # Reduced for Jupyter stability; increase to 300 (sparse), 500 (normal), 600 (dense) if stable
        batch_size = 128 if condition == 'dense' else 64 if condition == 'normal' else 32
        agent = MQLCAgent(
            obs_dim=7,
            state_dim=env.state_dim,
            action_dim=5,
            num_avs=num_avs,
            num_vehicles=env.num_vehicles,
            condition=condition,
            hidden_dim=64 if condition == 'dense' else 32,
            lr=0.0003 if condition == 'dense' else 0.0005,
            gamma=0.995,
            lambda_reg=0.05,
            epsilon=1.0,
            alpha=1.0
        )
        
        for episode in range(episodes):
            state, observations = env.reset()
            done = False
            step = 0
            while not done:
                step += 1
                m_velocity, t_density, s_variance = env.get_traffic_conditions(state)
                priorities = [agent.compute_priority(m_velocity, t_density, s_variance) for _ in range(num_avs)]
                env_features = torch.tensor([m_velocity, t_density, s_variance], dtype=torch.float32).to(device)
                obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
                
                x_positions = state[1::5].reshape(-1, 1)
                y_positions = state[2::5].reshape(-1, 1)
                x_dists = x_positions - x_positions.T
                x_dists = np.minimum(x_dists, env.road_length - x_dists)
                y_dists = y_positions - y_positions.T
                dists = np.sqrt(x_dists**2 + y_dists**2)
                adj = torch.tensor(dists < 10.0, dtype=torch.float32).to(device)
                adj.fill_diagonal_(0)
                adj = adj[:num_avs, :num_avs].unsqueeze(0)
                
                actions = agent.select_action(obs_tensor, state, adj, env_features, priorities)
                joint_action = agent.select_joint_action(torch.tensor(state, dtype=torch.float32).to(device), adj, env_features, actions)
                
                next_state, next_observations, rewards, global_reward, done = env.step(joint_action)
                
                for i, (act, rew) in enumerate(zip(actions, rewards)):
                    agent.store_transition(observations, act, rew, next_observations, state, joint_action, global_reward, next_state, i)
                
                if step % 20 == 0:
                    logger.info(f"Episode {episode + 1}, Step {step}, Reward={global_reward:.2f}, Collisions={env.episode_collisions}")
                
                agent.update(batch_size=batch_size)
                
                state, observations = next_state, next_observations
                
                if episode % 10 == 0 and step == 1 and psutil:
                    mem_info = psutil.Process().memory_info()
                    logger.info(f"Memory usage: {mem_info.rss / 1024**2:.2f} MB")
            
            if episode % 10 == 0:
                avg_reward = env.episode_rewards[-1] if env.episode_rewards else 0
                logger.info(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Length: {env.episode_lengths[-1] if env.episode_lengths else 60}, Avg Speed: {env.avg_speeds[-1] if env.avg_speeds else 0:.2f}, Collisions: {env.episode_collisions}")
                
                try:
                    sns.set(style="whitegrid", palette="husl")
                    plt.figure(figsize=(12, 10))
                    
                    plt.subplot(2, 2, 1)
                    plt.plot(env.episode_rewards, label='Average Reward', color='royalblue', linewidth=2.5)
                    plt.xlabel('Episode', fontsize=12)
                    plt.ylabel('Reward', fontsize=12)
                    plt.title(f'Average Reward per Episode ({condition})', fontsize=14, weight='bold')
                    plt.legend(fontsize=10)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(env.episode_lengths, label='Episode Length', color='seagreen', linewidth=2.5)
                    plt.xlabel('Episode', fontsize=12)
                    plt.ylabel('Steps', fontsize=12)
                    plt.title(f'Episode Length ({condition})', fontsize=14, weight='bold')
                    plt.legend(fontsize=10)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.subplot(2, 2, 3)
                    plt.plot(env.avg_speeds, label='Average Speed', color='coral', linewidth=2.5)
                    plt.xlabel('Episode', fontsize=12)
                    plt.ylabel('Speed (m/s)', fontsize=12)
                    plt.title(f'Average Speed per Episode ({condition})', fontsize=14, weight='bold')
                    plt.legend(fontsize=10)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.subplot(2, 2, 4)
                    collision_rates = [c / max(1, l) for c, l in zip(env.episode_collisions_list, env.episode_lengths)]
                    plt.plot(collision_rates, label='Collision Rate', color='purple', linewidth=2.5)
                    plt.xlabel('Episode', fontsize=12)
                    plt.ylabel('Collisions/Step', fontsize=12)
                    plt.title(f'Collision Rate per Episode ({condition})', fontsize=14, weight='bold')
                    plt.legend(fontsize=10)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(f'traffic_metrics_{condition}_ep{episode + 1}.png', dpi=300, bbox_inches='tight')
                    plt.close('all')
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(agent.ind_losses, label='Individual Q Loss', color='dodgerblue', linewidth=2)
                    plt.plot(agent.glo_losses, label='Global Q Loss', color='darkorange', linewidth=2)
                    plt.xlabel('Update Step', fontsize=12)
                    plt.ylabel('Loss', fontsize=12)
                    plt.title(f'Q-Network Losses ({condition})', fontsize=14, weight='bold')
                    plt.legend(fontsize=10)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.savefig(f'q_losses_{condition}_ep{episode + 1}.png', dpi=300, bbox_inches='tight')
                    plt.close('all')
                    logger.info(f"Plots saved as 'traffic_metrics_{condition}_ep{episode + 1}.png' and 'q_losses_{condition}_ep{episode + 1}.png'")
                except Exception as e:
                    logger.error(f"Error during plotting for {condition}: {e}")
                    raise e
            
            if episode % 10 == 0:
                for target_param, param in zip(agent.target_individual_q.parameters(), agent.individual_q.parameters()):
                    target_param.data.copy_(param.data)
                for target_param, param in zip(agent.target_global_q.parameters(), agent.global_q.parameters()):
                    target_param.data.copy_(param.data)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        collision_rate = env.collisions / max(1, env.total_steps)
        avg_length = np.mean(env.episode_lengths) if env.episode_lengths else 0
        avg_speed = np.mean(env.avg_speeds) if env.avg_speeds else 0
        avg_reward = np.mean(env.episode_rewards) if env.episode_rewards else 0
        
        logger.info(f"\nFinal Results for {condition}:")
        logger.info(f"Collision Rate: {collision_rate:.4f} collisions/step")
        logger.info(f"Average Episode Length: {avg_length:.2f} steps")
        logger.info(f"Average Speed: {avg_speed:.2f} m/s")
        logger.info(f"Average Reward: {avg_reward:.2f}")
        
        logger.info(f"Training completed for {condition}")
        logger.removeHandler(fh)
        fh.close()
        
        return {
            'condition': condition,
            'avg_length': avg_length,
            'avg_speed': avg_speed,
            'avg_reward': avg_reward
        }
    except Exception as e:
        logger.error(f"Error in train_mqlc_for_condition {condition}: {e}")
        raise e
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_all_conditions():
    conditions = [
        {'num_avs': 2, 'num_hdvs': 8, 'name': 'sparse'},
        {'num_avs': 3, 'num_hdvs': 12, 'name': 'normal'},
        {'num_avs': 4, 'num_hdvs': 24, 'name': 'dense'}
    ]
    
    results = []
    
    for cond in conditions:
        result = train_mqlc_for_condition(cond['num_avs'], cond['num_hdvs'], cond['name'])
        results.append(result)
    
    print("\nFinal Results Summary:")
    for result in results:
        cond = result['condition']
        print(f"\n{cond.capitalize()}:")
        print(f"Length: {result['avg_length']:.2f} steps")
        print(f"Avg Speed: {result['avg_speed']:.2f} m/s")
        print(f"Reward: {result['avg_reward']:.2f}")

if __name__ == "__main__":
    try:
        run_all_conditions()
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise e
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()