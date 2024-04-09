import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque

from environment import Environment

# Environment class with state update function



# DQN model


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.network(state)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


# Function to select an action using epsilon-greedy policy
def select_action(state, policy_net, epsilon, n_actions):
    if random.random() > epsilon:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = policy_net(state)
        action = action_values.max(1)[1].item()
    else:
        action = random.randrange(n_actions)
    return action


# Function to perform a single step of optimization
def optimize_model(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to tensors
    state_batch = torch.FloatTensor(states).to(device)
    action_batch = torch.LongTensor(actions).view(-1, 1).to(device)
    reward_batch = torch.FloatTensor(rewards).to(device)
    next_state_batch = torch.FloatTensor(next_states).to(device)
    done_batch = torch.FloatTensor(dones).to(device)

    # Compute current Q values
    current_q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute next Q values from target network
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    # Compute loss
    loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Training hyperparameters
state_size = 100  # Adjusted based on environment's state size
action_size = 20  # Number of actions
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.0
epsilon_decay = 300
learning_rate = 1e-3
target_update = 10

# Initialize components
env = Environment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set target_net to evaluation mode
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(1000000)
epsilon = epsilon_start
rewards = [] 
nodes = {}
losses = []
# Training loop

num_episodes = 1000
num_tasks = 300
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    for i in range(num_tasks):
        action = select_action(state, policy_net, epsilon, action_size)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        # print('Total reward', total_reward)
        loss = optimize_model(
            policy_net, target_net, optimizer, replay_buffer, batch_size, gamma
        )

        if loss is not None:  # Make sure to only append when loss is returned
            losses.append(loss)

        if action in nodes:
            nodes[action] += 1
        else:
            nodes[action] = 1

    epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

        
    rewards.append(total_reward)
    print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon}")
print('end loop')
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards vs. Episodes')
plt.legend()
plt.show()

node_numbers = list(nodes.keys())
frequencies = list(nodes.values())

# Creating the bar chart
plt.figure(figsize=(10, 6))  # Optional: Adjusts the size of the chart
plt.bar(node_numbers, frequencies, color='skyblue')  # Creates the bar chart with skyblue bars

plt.xlabel('Node Number')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.title('Frequency of Nodes from 0 to 20')  # Title of the chart
plt.xticks(node_numbers)  # Ensures each node number is marked on the x-axis
plt.grid(axis='y', linestyle='--')  # Optional: Adds a grid for the y-axis

plt.show()  # Displays the chart
# Remember to fill in the missing parts of the environment and adjust hyperparameters as necessary

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss per Optimization Step')
plt.xlabel('Optimization Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

