import traci
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import pickle

# Disable traci and SUMO logging
traci_log = logging.getLogger('traci')
traci_log.setLevel(logging.CRITICAL)  # Suppress logging from traci (only critical errors will be shown)

sumo_log = logging.getLogger('sumo')
sumo_log.setLevel(logging.CRITICAL)  # Suppress SUMO logs

# Hyperparameters
GAMMA = 0.95  # Discount factor
LR = 0.005  # Learning rate
BATCH_SIZE_START = 32  # Initial batch size
BATCH_SIZE_MAX = 128  # Maximum batch size
MEMORY_SIZE = 10000
TARGET_UPDATE = 10  # Update target network every X steps
EPISODES = 10000  # Number of episodes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON_START = 1.0  # Start with full exploration
EPSILON_END = 0.01  # End with minimal exploration
EPSILON_DECAY = 0.0001  # Decay rate per episode
EPSILON = EPSILON_START

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
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.int64, device=DEVICE),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                with open(filename, 'rb') as f:
                    self.buffer = deque(pickle.load(f), maxlen=self.buffer.maxlen)
            except EOFError:
                print(f"Error: The file {filename} is empty or corrupted.")
        else:
            print(f"Error: The file {filename} does not exist or is empty.")

def connect_to_sumo(sumo_binary="sumo", config_file="./simulation.sumocfg"):
    sumo_cmd = [
        sumo_binary,
        "-c", config_file,
        "--collision.action", "none",
        "--time-to-teleport", "-1",
        "--no-step-log",
         "--no-warnings"
    ]
    traci.start(sumo_cmd)

FIXED_STATE_DIM = 60

def get_state():
    state = []
    traffic_lights = traci.trafficlight.getIDList()
    for tl in traffic_lights:
        state.extend([int(c == 'G') for c in traci.trafficlight.getRedYellowGreenState(tl)])
    
    vehicles = traci.vehicle.getIDList()
    for veh in vehicles:
        state.extend(traci.vehicle.getPosition(veh))
        state.append(traci.vehicle.getSpeed(veh))
    
    state = np.array(state, dtype=np.float32)
    
    if len(state) > FIXED_STATE_DIM:
        state = state[:FIXED_STATE_DIM]
    elif len(state) < FIXED_STATE_DIM:
        state = np.pad(state, (0, FIXED_STATE_DIM - len(state)), mode='constant')
    
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

def choose_action_BAK(state, model, action_dim, epsilon, current_phase, previous_phase):
    if random.uniform(0, 1) < epsilon:
        valid_actions = get_valid_actions(current_phase, previous_phase, action_dim)
        return random.choice(valid_actions)

    state = np.array([state])
    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        q_values = model(state_tensor)
    
    valid_actions = get_valid_actions(current_phase, previous_phase, action_dim)
    q_values = q_values.cpu().numpy().flatten()
    valid_q_values = [q_values[action] for action in valid_actions]
    best_action = valid_actions[np.argmax(valid_q_values)]
    
    return best_action

def get_valid_actions(current_phase, previous_phase, action_dim):
    if current_phase == 0:
        return [1]
    elif current_phase == 1:
        if previous_phase == 0:
            return [2]
        elif previous_phase == 2:
            return [0]
    elif current_phase == 2:
        return [1]
    return list(range(action_dim))

def compute_reward():
    collisions = traci.simulation.getCollidingVehiclesIDList()
    if collisions:
        print(f"Collision detected! Vehicles involved: {collisions}")
        return -10000
    
    vehicles = traci.vehicle.getIDList()
    if not vehicles:
        return 0, 0, 0
    
    total_speed = sum(traci.vehicle.getSpeed(veh) for veh in vehicles)
    total_waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
    
    avg_speed = total_speed / len(vehicles)
    avg_waiting_time = total_waiting_time / len(vehicles)
    
    w1 = 1.0
    w2 = 0.01
    reward = (w1 * avg_speed) - (w2 * avg_waiting_time)
    
    return reward, avg_speed, avg_waiting_time

def train_model(policy_net, target_net, replay_buffer, optimizer, BATCH_SIZE):
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_traffic_light_phase_counts():
    traffic_lights = traci.trafficlight.getIDList()
    phase_counts = []
    for tl in traffic_lights:
        program_logics = traci.trafficlight.getAllProgramLogics(tl)
        phase_count = sum(len(logic.phases) for logic in program_logics)
        phase_counts.append(phase_count)
    return phase_counts

def main(EPSILON = EPSILON_START, BATCH_SIZE = BATCH_SIZE_START):
    connect_to_sumo()

    state_dim = FIXED_STATE_DIM
    print(f"Running training for {EPISODES} episodes...")

    traffic_lights = traci.trafficlight.getIDList()
    phase_counts = get_traffic_light_phase_counts()
    action_dim = sum(phase_counts)

    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    
    if os.path.exists("replay_buffer.npy"):
        replay_buffer.load("replay_buffer.npy")
        print("Replay buffer loaded.")

    if os.path.exists("dqn_traffic_model.pth"):
        policy_net.load_state_dict(torch.load("dqn_traffic_model.pth"))
        target_net.load_state_dict(policy_net.state_dict())
        print("Model loaded.")

    EPSILON = EPSILON_START

    for episode in range(EPISODES):
        EPSILON = max(EPSILON_END, EPSILON - EPSILON_DECAY)
        if (episode + 1) % 100 == 0:
            BATCH_SIZE = min(BATCH_SIZE_MAX, BATCH_SIZE + 1)
        traci.load(["-c", "./simulation.sumocfg"])
        state = get_state()

        total_reward = 0
        total_avg_speed = 0
        total_avg_waiting_time = 0
        steps_in_episode = 0


        for step in range(1000):

            #action = choose_action(state, policy_net, action_dim, EPSILON, current_phase, previous_phase)
            action = choose_action(state, policy_net, action_dim, EPSILON)

            collisions = traci.simulation.getCollidingVehiclesIDList()
            if collisions:
                print(f"Collision detected at step {step}. Ending episode early.")
                reward = -10000
                done = True
                replay_buffer.push(state, action, reward, state, done)
                break
            

            if traci.simulation.getMinExpectedNumber() <= 0:
                print("No more vehicles in simulation. Ending early.")
                break

            traci.simulationStep()

            action_offset = 1
            for i, tl in enumerate(traffic_lights):
                num_phases = phase_counts[i]
                if num_phases > 0:
                    phase_action = (action // action_offset) % num_phases
                    traci.trafficlight.setPhase(tl, phase_action)
                    action_offset *= num_phases

            next_state = get_state()
            reward, avg_speed, avg_waiting_time = compute_reward()

            total_reward += reward
            total_avg_speed += avg_speed
            total_avg_waiting_time += avg_waiting_time
            steps_in_episode += 1

            done = step == 999

            replay_buffer.push(state, action, reward, next_state, done)

            train_model(policy_net, target_net, replay_buffer, optimizer, BATCH_SIZE)

            state = next_state

        avg_reward = total_reward / steps_in_episode if steps_in_episode > 0 else 0
        avg_speed = total_avg_speed / steps_in_episode if steps_in_episode > 0 else 0
        avg_waiting_time = total_avg_waiting_time / steps_in_episode if steps_in_episode > 0 else 0

        print(f"Episode {episode+1} - Avg Reward: {avg_reward:.2f}, Avg Speed: {avg_speed:.2f} m/s, Avg Waiting Time: {avg_waiting_time:.2f} s")

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 100 == 0 or (episode + 1) == EPISODES:
            save_model(policy_net, f"dqn_traffic_model_episode_{episode + 1}.pth")
            replay_buffer.save("replay_buffer.pkl")

    traci.close()

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    main()