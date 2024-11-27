import traci
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Disable traci and SUMO logging
traci_log = logging.getLogger('traci')
traci_log.setLevel(logging.CRITICAL)  # Suppress logging from traci (only critical errors will be shown)

sumo_log = logging.getLogger('sumo')
sumo_log.setLevel(logging.CRITICAL)  # Suppress SUMO logs

# Hyperparameters
GAMMA = 0.95  # Discount factor
LR = 0.005  # Learning rate
BATCH_SIZE = 128
MEMORY_SIZE = 10000
TARGET_UPDATE = 10  # Update target network every X steps
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
        # Sample directly from the buffer using numpy for efficient indexing
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to numpy arrays before converting to tensors
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Convert numpy arrays to torch tensors efficiently
        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.int64, device=DEVICE),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

def connect_to_sumo(sumo_binary="sumo", config_file="./simulation.sumocfg"):
    sumo_cmd = [
    "sumo",
    "-c", "simulation.sumocfg",
    "--collision.action", "none",  # No teleporting on collisions
    "--time-to-teleport", "-1",  # Disable teleporting for blocked vehicles
    "--no-step-log",  # Suppress log spam
    "--no-warnings"  # Suppress warnings
    ]
    traci.start(sumo_cmd)

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
    
    return state


def choose_action(state, model, action_dim, epsilon):
    """
    Chooses an action using epsilon-greedy policy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_dim - 1)  # Random action

    # Wrap state in a NumPy array and convert to tensor
    state = np.array([state])  # Ensure it's a 2D array
    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        q_values = model(state_tensor)
    return q_values.argmax().item()  # Best action

def compute_reward():
    """
    Computes a reward for the current simulation state, including average speed, 
    average waiting time, and the reward based on these metrics.
    """

    # Check for collisions in the current step
    collisions = traci.simulation.getCollidingVehiclesIDList()
    if collisions:  # If there are any collisions
        print(f"Collision detected! Vehicles involved: {collisions}")
        return -10000  # Assign a heavy penalty for collisions
    
    vehicles = traci.vehicle.getIDList()
    if not vehicles:  # If no vehicles in the simulation, avoid division by zero
        return 0, 0, 0  # Reward, avg_speed, avg_waiting_time
    
    total_speed = sum(traci.vehicle.getSpeed(veh) for veh in vehicles)
    total_waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
    
    avg_speed = total_speed / len(vehicles)
    avg_waiting_time = total_waiting_time / len(vehicles)
    
    # Balance avg_speed and avg_waiting_time using weights
    w1 = 1.0
    w2 = 0.01  # Scale down waiting time's influence
    reward = (w1 * avg_speed) - (w2 * avg_waiting_time)
    
    return reward, avg_speed, avg_waiting_time

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
    traffic_lights = traci.trafficlight.getIDList()
    phase_counts = []
    for tl in traffic_lights:
        program_logics = traci.trafficlight.getAllProgramLogics(tl)
        # Assuming you're interested in the number of phases in the first program logic
        # You can iterate over multiple program logics if necessary
        phase_count = sum(len(logic.phases) for logic in program_logics)
        phase_counts.append(phase_count)
    return phase_counts


def main():
    connect_to_sumo()

    state_dim = FIXED_STATE_DIM  # Assume state_dim is defined elsewhere
    print(f"Running training for {EPISODES} episodes...")

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
    
    EPSILON_START = 1.0  # Start with full exploration
    EPSILON_END = 0.1  # End with minimal exploration
    EPSILON_DECAY = 0.0009  # Decay rate per episode
    EPSILON = EPSILON_START

    for episode in range(EPISODES):
        EPSILON = max(EPSILON_END, EPSILON - EPSILON_DECAY)  # Ensure it doesn't go below EPSILON_END
        print(f"Episode {episode+1}/{EPISODES}: Epsilon = {EPSILON}")
        traci.load(["-c", "./simulation.sumocfg"])  # Reload the simulation for each episode
        state = get_state()

        total_reward = 0
        total_avg_speed = 0
        total_avg_waiting_time = 0
        steps_in_episode = 0

        for step in range(1000):  # Adjust as per simulation duration
            action = choose_action(state, policy_net, action_dim, EPSILON)

            collisions = traci.simulation.getCollidingVehiclesIDList()
            #print(f"Collisions: {len(collisions)}")
            if collisions:  # If there are collisions
                print(f"Collision detected at step {step}. Ending episode early.")
                reward = -10000  # Assign minimum reward (tune as necessary)
                done = True
                replay_buffer.push(state, action, reward, state, done)  # Store terminal state
                break
                
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("No more vehicles in simulation. Ending early.")
                break
                
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
            reward, avg_speed, avg_waiting_time = compute_reward()

            # Accumulate metrics for logging
            total_reward += reward
            total_avg_speed += avg_speed
            total_avg_waiting_time += avg_waiting_time
            steps_in_episode += 1

            done = step == 999  # End episode after fixed steps

            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Train the model
            train_model(policy_net, target_net, replay_buffer, optimizer)

            state = next_state

        # Log the average statistics for the episode
        avg_reward = total_reward / steps_in_episode if steps_in_episode > 0 else 0
        avg_speed = total_avg_speed / steps_in_episode if steps_in_episode > 0 else 0
        avg_waiting_time = total_avg_waiting_time / steps_in_episode if steps_in_episode > 0 else 0
        
        print(f"Episode {episode+1} - Avg Reward: {avg_reward:.2f}, Avg Speed: {avg_speed:.2f} m/s, Avg Waiting Time: {avg_waiting_time:.2f} s")

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save model every 100 episodes
        if (episode + 1) % 100 == 0 or (episode + 1) == EPISODES:
            save_model(policy_net, f"dqn_traffic_model_episode_{episode + 1}.pth")


    traci.close()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    main()

