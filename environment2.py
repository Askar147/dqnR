import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque

class Environment2:
    def __init__(self):
        self.n_nodes = 20
        self.state = np.zeros((self.n_nodes, 4))  # PowerUsage, AllocatedCPUn, FreeMemoryn, Ranking
        self.state[:, 2] = 2000  # Initialize FreeMemoryn to 2000
        self.max_power = 5000  # Max power consumption
        self.cpu_frequency = 1000  # CPU Frequency in MHz
        self.max_cores = 4  # Maximum number of CPU cores
        self.step_count = 0

    def reset(self):
        self.state = np.zeros((self.n_nodes, 4))
        self.state[:, 2] = 2000
        self.step_count = 0
        return self.state.flatten()

    def step(self, action):
        self.step_count += 1
        self.manage_tasks(action)
        reward = self.calculate_reward()
        next_state = self.state.flatten()
        done = False  # Define your condition for episode termination
        return next_state, reward, done

    def manage_tasks(self, action):
        # Task arrival
        required_cpu_cycles = np.random.randint(50, 101)  # Required CPU Cycles to Compute Task T
        required_memory = np.random.randint(64, 257)  # Required Memory in MByte
        vcpu_for_docker = np.random.randint(20, 61)  # VCPU for Docker in ms
        user_deadline = np.random.randint(30, 241)  # Deadline by User in sec

        # Update state based on action and task properties
        if self.can_allocate(action):
            # Calculate actual deadline if allocated CPU is non-zero

            cpu_load = 1 - min(max(self.state[action][1] / 100, 0), 1)

            actual_deadline = required_cpu_cycles / (cpu_load * self.cpu_frequency)
            # Update FreeMemory
            self.state[action][2] -= required_memory
            # Update AllocatedCPUn
            self.state[action][1] += vcpu_for_docker / self.max_cores
            # Calculate power usage
            power_usage = self.max_power * self.state[action][1] / (user_deadline / actual_deadline)
            # Update PowerUsage
            self.state[action][0] = power_usage

    def can_allocate(self, node_index):
        return self.state[node_index][2] > 0 and self.state[node_index][1] < 100

    def calculate_reward(self):
        # Calculate the total power usage
        total_power_usage = np.sum(self.state[:, 0])
        
        # Normalize the total power usage to range from 0 to -1
        # Here, we assume that the maximum power usage is 5000 (you may need to adjust this value based on your actual scenario)
        normalized_reward = -total_power_usage / 5000
        
        return normalized_reward

    def update_ranking(self):
        # Update ranking based on available resources (higher is better)
        for i in range(self.n_nodes):
            # Higher ranking for more free memory
            self.state[i][3] = self.state[i][2]
            # Higher ranking for lower allocated CPU percentage (more free CPU)
            self.state[i][3] += (100 - self.state[i][1])
            # Higher ranking for lower power usage
            self.state[i][3] += (self.max_power - self.state[i][0])