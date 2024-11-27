import traci
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 0.001  # Learning rate
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10  # Update target network every X steps
EPSILON = 0.1  # Exploration rate
EPISODES = 1000  # Number of episodes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network for DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.int64, device=DEVICE),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

def connect_to_sumo(sumo_binary="sumo-gui", config_file="your_simulation.sumocfg"):
    traci.start([sumo_binary, "-c", config_file])

def get_state():
    """
    Creates a compact state representation for the simulation.
    """
    state = []
    # Include traffic light states and vehicle positions/speeds
    traffic_lights = traci.trafficlight.getIDList()
    for tl in traffic_lights:
        state.extend([int(c == 'G') for c in traci.trafficlight.getRedYellowGreenState(tl)])  # Encode RYG as binary

    vehicles = traci.vehicle.getIDList()
    for veh in vehicles:
        state.extend(traci.vehicle.getPosition(veh))
        state.append(traci.vehicle.getSpeed(veh))
    
    return np.array(state, dtype=np.float32)

def choose_action(state, model, action_dim):
    """
    Chooses an action using epsilon-greedy policy.
    """
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, action_dim - 1)  # Random action
    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    return q_values.argmax().item()  # Best action

def compute_reward():
    """
    Computes a reward for the current simulation state.
    """
    vehicles = traci.vehicle.getIDList()
    total_speed = sum(traci.vehicle.getSpeed(veh) for veh in vehicles)
    total_waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
    reward = total_speed - total_waiting_time  # Encourage speed, discourage waiting
    return reward

def train_model(policy_net, target_net, replay_buffer, optimizer):
    """
    Trains the policy network using experiences from the replay buffer.
    """
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Compute Q-values for current states
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    connect_to_sumo(config_file="path/to/your_simulation.sumocfg")

    # Initialize DQN components
    state_dim = len(get_state())
    action_dim = len(traci.trafficlight.getIDList())
    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    for episode in range(EPISODES):
        print(f"Episode {episode+1}/{EPISODES}")
        traci.load(["-c", "path/to/your_simulation.sumocfg"])  # Reload the simulation for each episode
        state = get_state()

        for step in range(1000):  # Adjust as per simulation duration
            action = choose_action(state, policy_net, action_dim)
            traci.simulationStep()

            # Apply action (e.g., change traffic light phases)
            traffic_lights = traci.trafficlight.getIDList()
            for i, tl in enumerate(traffic_lights):
                traci.trafficlight.setPhase(tl, action % len(traci.trafficlight.getControlledLinks(tl)))

            # Get next state and reward
            next_state = get_state()
            reward = compute_reward()
            done = step == 999  # End episode after fixed steps

            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Train the model
            train_model(policy_net, target_net, replay_buffer, optimizer)

            state = next_state

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    traci.close()

if __name__ == "__main__":
    main()

