import random
import collections
import math
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from env import ConnectFourEnv, ReplayBuffer

def select_action(net, state, epsilon, n_actions, device, env):
    if random.random() < epsilon:
        return random.choice(env.valid_columns)
    state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    qvals = net(state_v)

    for c in env.invalid_columns:
        qvals[0, c] = float('-inf')

    return int(torch.argmax(qvals, dim=1).item())

def compute_td_loss(batch, policy_net, target_net, device, gamma):
    states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    expected_q = rewards + gamma * next_q_values * (1.0 - dones)
    return F.mse_loss(q_values, expected_q)


# Training loop
def train(
    num_episodes=5000,
    max_steps_per_episode=42,
    buffer_capacity=50000,
    batch_size=64,
    gamma=0.99,
    lr=0.001,
    hidden_dim=128,
    target_update_freq=1000,
    start_training_after=60,
    train_freq=4,
    eps_start=1.0,
    eps_final=0.02,
    eps_decay=10000,
    save_path="connect_four",
    load_path=None,
    render_every_n=0,
    switch_agent_train = 2000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConnectFourEnv(render=False)
    n_actions = 7

    total_steps = 0

    # agent 1
    policy_net_1 = QNetwork(hidden_dim, n_actions).to(device)
    if load_path is not None:
        policy_net_1.load_state_dict(torch.load(f"{load_path}_agent_1"))
    target_net_1 = QNetwork(hidden_dim, n_actions).to(device)
    target_net_1.load_state_dict(policy_net_1.state_dict())
    target_net_1.eval()
    optimizer_1 = optim.Adam(policy_net_1.parameters(), lr=lr)
    replay_1 = ReplayBuffer(capacity=buffer_capacity)
    episode_rewards_1 = []

    # agent 2
    policy_net_2 = QNetwork(hidden_dim, n_actions).to(device)
    if load_path is not None:
        policy_net_2.load_state_dict(torch.load(f"{load_path}_agent_2"))
    target_net_2 = QNetwork(hidden_dim, n_actions).to(device)
    target_net_2.load_state_dict(policy_net_2.state_dict())
    target_net_2.eval()
    optimizer_2 = optim.Adam(policy_net_2.parameters(), lr=lr)
    replay_2 = ReplayBuffer(capacity=buffer_capacity)
    episode_rewards_2 = []

    if render_every_n > 0:
        env_render = ConnectFourEnv(render=True)
    else:
        env_render = None

    for eps in range(num_episodes):
        state = env.reset()
        ep_reward_1 = 0.0
        ep_reward_2 = 0.0
        for step in range(max_steps_per_episode):
            total_steps += 1
            if total_steps % 2 == 0:
                epsilon = eps_final + (eps_start - eps_final) * math.exp(-1.0 * total_steps / eps_decay)
                action = select_action(policy_net_1, state, epsilon, n_actions, device, env)
                next_state, reward, done, _ = env.step(action)
                ep_reward_1 += reward
                replay_1.push(state, action, reward, next_state, float(done))
                state = next_state

                if len(replay_1) > batch_size and total_steps > start_training_after and total_steps % train_freq == 0 and total_steps % switch_agent_train >= switch_agent_train * 0.5:
                    batch = replay_1.sample(batch_size)
                    loss = compute_td_loss(batch, policy_net_1, target_net_1, device, gamma)
                    optimizer_1.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net_1.parameters(), 1.0)
                    optimizer_1.step()

                if total_steps % target_update_freq == 0:
                    target_net_1.load_state_dict(policy_net_1.state_dict())
            else:
                epsilon = eps_final + (eps_start - eps_final) * math.exp(-1.0 * total_steps / eps_decay)
                action = select_action(policy_net_2, state, epsilon, n_actions, device, env)
                next_state, reward, done, _ = env.step(action)
                ep_reward_2 += reward
                replay_2.push(state, action, reward, next_state, float(done))
                state = next_state

                if len(replay_2) > batch_size and total_steps > start_training_after and total_steps % train_freq == 0 and total_steps % switch_agent_train < switch_agent_train * 0.5:
                    batch = replay_2.sample(batch_size)
                    loss = compute_td_loss(batch, policy_net_2, target_net_2, device, gamma)
                    optimizer_2.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net_2.parameters(), 1.0)
                    optimizer_2.step()

                if total_steps % target_update_freq == 0:
                    target_net_2.load_state_dict(policy_net_2.state_dict())

            # Render
            if env_render is not None and eps % render_every_n == 0:
                env_render.grid = env.grid
                env_render.render()
            if done:
                break

        episode_rewards_1.append(ep_reward_1)
        episode_rewards_2.append(ep_reward_2)
        if eps % 10 == 0:
            avg_reward_1 = np.mean(episode_rewards_1[-100:])
            avg_reward_2 = np.mean(episode_rewards_2[-100:])
            print(f"Episode {eps}  TotalSteps {total_steps}  Epsilon {epsilon:.3f}  AverageReward for agent 1 {avg_reward_1:3f}")
            print(f"Episode {eps}  TotalSteps {total_steps}  Epsilon {epsilon:.3f}  AverageReward for agent 2 {avg_reward_2:3f}")

        # Save Model
        if eps % 20000 == 0:
            torch.save(policy_net_1.state_dict(), f"{save_path}_agent_1_{eps}")
            torch.save(policy_net_2.state_dict(), f"{save_path}_agent_2_{eps}")

    if env_render is not None:
        env_render.close()
    env.close()

train(
    num_episodes=300001,
    batch_size = 64,
    lr = 0.0001,
    gamma = 0.99,
    eps_start = 1.0,
    eps_final = 0.15,
    eps_decay = 3000000,
    render_every_n = 3000
)



