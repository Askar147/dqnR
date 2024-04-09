import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque

# Environment class with state update function
class Environment:
    def __init__(self):
        self.n_nodes = 20
        self.state = np.zeros((self.n_nodes, 5))
        # Initialize free and available memory randomly between 1600 and 2000
        self.state[:, 2:4] = np.random.randint(1900, 2001, size=(self.n_nodes, 2))
        # Initialize power usage randomly between 800 and 1000
        self.state[:, 4] = np.random.randint(800, 1001, size=self.n_nodes)

        self.task_queue = {}  # Tasks waiting to be allocated
        self.task_tracker = {}  # Active tasks: {task_id: (node_index, steps_remaining)}
        self.task_deadline = 120  # Deadline in steps for tasks
        self.step_count = 0

    def reset(self):
        self.state = np.zeros((self.n_nodes, 5))
        # Initialize free and available memory randomly between 1600 and 2000
        self.state[:, 2:4] = np.random.randint(1900, 2001, size=(self.n_nodes, 2))
        # Initialize power usage randomly between 800 and 1000
        self.state[:, 4] = np.random.randint(800, 1001, size=self.n_nodes)
        self.task_queue = {}
        self.task_tracker = {}
        self.step_count = 0
        return self.state.flatten()

    def step(self, action):

        self.required_resource = np.random.randint(64, 128)

        self.step_count += 1
        self.manage_tasks(action)
        self.check_task_completion()
        reward = self.calculate_reward()
        next_state = self.state.flatten()
        done = False  # Define your condition for episode termination
        return next_state, reward, done

    def manage_tasks(self, action):
        # Check if the node can allocate resources for a new task
        if self.can_allocate(action):
            self.allocate_resources(action)
        else:
            self.add_task_to_queue(action)
        
        # Update tasks in queue and apply penalties
        self.update_queue()


    def can_allocate(self, node_index):
        # Implement logic to check if node can handle a new task
        # Example: Check if cpuload and allocatedcpu are below a threshold
        return self.state[node_index, 0] <= 90 and self.state[node_index, 1] < 90 and self.state[node_index, 2] > 100
    

    #TODO: IMPROVE RESOURCE ALLOCATION: 1: allocate them differently 2: track them
    #TODO: SAVE ALLOCATED RESOURCES
    def allocate_resources(self, node_index):

        # Here, add the task to task_tracker with its execution time
        task_id = self.generate_task_id()  # Implement this method to generate unique task IDs
        self.task_tracker[task_id] = (node_index, self.task_deadline)

        # Allocate resources as before in update_state, modified for clarity
        # Ensure to adjust based on your actual resource allocation logic
        cpu_reduction = np.random.randint(8, 14)
        self.state[node_index, 0] = self.state[node_index, 0] + cpu_reduction # cpuload

        self.state[node_index, 1] = min(
            self.state[node_index, 1] + 10, 90
        )  # allocatedCPU
        memory_reduction = np.random.randint(80, 161)
        self.state[node_index, 2] = max(
            self.state[node_index, 2] - memory_reduction, 0
        )  # freememory
        self.state[node_index, 3] = max(
            self.state[node_index, 3] - memory_reduction, 0
        )  # availablememory

        # Assuming a simplified relationship between cpuload and power usage for demonstration
        #TODO: 10 is for cpuload change. Needs to change more interestingly
        self.state[node_index, 4] = (
            5000 * self.state[node_index, 0] / 10 * self.required_resource / self.state[node_index, 2]
        )   # Adjust based on your model or assumptions


    def generate_task_id(self):
        # Generate a unique task ID (simple implementation)
        return len(self.task_tracker) + 1

    def check_task_completion(self):
        # Check each task in task_tracker for completion
        completed_tasks = []
        for task_id, (node_index, steps_remaining) in self.task_tracker.items():
            steps_remaining -= 1
            if steps_remaining <= 0:
                completed_tasks.append(task_id)
                self.free_resources(node_index)  # Implement based on your resource freeing logic
            else:
                self.task_tracker[task_id] = (node_index, steps_remaining)

        # Remove completed tasks from tracker
        for task_id in completed_tasks:
            del self.task_tracker[task_id]

    def add_task_to_queue(self, action):
        if action in self.task_queue:
            self.task_queue[action] += 1
        else:
            self.task_queue[action] = 1

    def update_queue(self):
        # Penalize tasks staying in the queue
        for task, time_in_queue in self.task_queue.items():
            self.task_queue[task] += 1  # Increment time in queue

    #TODO: Implement partial resource free
    def free_resources(self, node_index):
        # Free up resources for tasks that have met their deadline
        # This is a placeholder; implement according to your resource management strategy
        self.state[node_index, 0] = max (self.state[node_index, 0] - 11,  0 ) # cpuload

        self.state[node_index, 1] = self.state[node_index, 1] - 11  # allocatedCPU

        memory_reduction = np.random.randint(80, 161)
        self.state[node_index, 2] = self.state[node_index, 2] + memory_reduction # freememory
        self.state[node_index, 3] = self.state[node_index, 3] + memory_reduction # allocatedmemory

        self.state[node_index, 4] = (
            5000 * self.state[node_index, 0] / 10 * self.required_resource / self.state[node_index, 2]
        )    # Adjust based on your model or assumptions

    def calculate_reward(self):
        # Calculate and return the reward, including penalties for queued tasks
        penalty = sum(self.task_queue.values()) * 10 # Example penalty
        penalty = len(self.task_queue.values()) * 1000
        reward = -np.sum(self.state[:, 4]) / 20
        return reward 
