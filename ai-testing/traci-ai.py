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

def connect_to_sumo(sumo_binary="sumo-gui", config_file="./simulation.sumocfg"):
    traci.start([sumo_binary, "-c", config_file])

FIXED_STATE_DIM = 60  # The fixed state dimension used throughout the code

def get_state():
    """
    Creates a fixed-size state representation for the simulation.
    """
    state = []
    # Include traffic light states
    traffic_lights = traci.trafficlight.getIDList()
    for tl in traffic_lights:
        state.extend([int(c == 'G') for c in traci.trafficlight.getRedYellowGreenState(tl)])  # Encode RYG as binary
    
    # Include vehicle positions and speeds
    vehicles = traci.vehicle.getIDList()
    for veh in vehicles:
        state.extend(traci.vehicle.getPosition(veh))
        state.append(traci.vehicle.getSpeed(veh))
    
    # Convert state to numpy array
    state = np.array(state, dtype=np.float32)
    
    # Pad or truncate state to ensure a fixed size
    if len(state) > FIXED_STATE_DIM:
        state = state[:FIXED_STATE_DIM]  # Truncate if too large
    elif len(state) < FIXED_STATE_DIM:
        state = np.pad(state, (0, FIXED_STATE_DIM - len(state)), mode='constant')  # Pad with zeros if too small
    
    print(f"State vector shape (padded/truncated): {state.shape}")  # Debugging
    return state


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
    print("Fetching vehicle IDs...")
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
    
def get_traffic_light_phase_counts():
    """
    Returns a list of the number of phases for each traffic light.
    """
    traffic_lights = traci.trafficlight.getIDList()
    return [len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0].phases) for tl in traffic_lights]


def main():
    connect_to_sumo(config_file="./simulation.sumocfg")

    # Fetch a sample state to determine the state dimension
    sample_state = get_state()
    print(f"Sample state (first call): {sample_state}")  # Debugging print
    state_dim = FIXED_STATE_DIM  # Use the fixed size
    print(f"Determined state dimension: {state_dim}")

    # Calculate action dimension
    traffic_lights = traci.trafficlight.getIDList()
    phase_counts = get_traffic_light_phase_counts()  # Get valid phase counts for all traffic lights
    action_dim = sum(phase_counts)  # Total actions across all traffic lights

    # Initialize neural network
    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    for episode in range(EPISODES):
        print(f"Episode {episode+1}/{EPISODES}")
        traci.load(["-c", "./simulation.sumocfg"])  # Reload the simulation for each episode
        state = get_state()

        for step in range(1000):  # Adjust as per simulation duration
            action = choose_action(state, policy_net, action_dim)
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("No more vehicles in simulation. Ending early.")
                break
            print("Fetching simulation step...")
            traci.simulationStep()

            # Apply actions to traffic lights
            action_offset = 1
            for i, tl in enumerate(traffic_lights):
                num_phases = phase_counts[i]
                if num_phases > 0:
                    phase_action = (action // action_offset) % num_phases
                    traci.trafficlight.setPhase(tl, phase_action)
                    action_offset *= num_phases

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

