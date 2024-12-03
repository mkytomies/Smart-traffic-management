import traci
import numpy as np
import torch
import torch.nn as nn
import logging

# Disable traci and SUMO logging
traci_log = logging.getLogger('traci')
traci_log.setLevel(logging.CRITICAL)  # Suppress logging from traci (only critical errors will be shown)

sumo_log = logging.getLogger('sumo')
sumo_log.setLevel(logging.CRITICAL)  # Suppress SUMO logs

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_STATE_DIM = 60  # The fixed state dimension used throughout the code

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

def connect_to_sumo(sumo_binary="sumo-gui", config_file="./simulation.sumocfg"):
    sumo_gui = [
        sumo_binary,
        "-c", config_file,
        "--collision.action", "none",  # No teleporting on collisions
        "--time-to-teleport", "-1",  # Disable teleporting for blocked vehicles
        "--no-step-log",  # Suppress log spam
        "--no-warnings"  # Suppress warnings
    ]
    traci.start(sumo_gui)

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

def choose_action(state, model, action_dim):
    """
    Chooses an action using the trained model.
    """
    # Wrap state in a NumPy array and convert to tensor
    state = np.array([state])  # Ensure it's a 2D array
    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        q_values = model(state_tensor)
    return q_values.argmax().item()  # Best action

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
    print("Running simulation with trained model...")

    # Calculate action dimension
    traffic_lights = traci.trafficlight.getIDList()
    phase_counts = get_traffic_light_phase_counts()  # Get valid phase counts for all traffic lights
    action_dim = sum(phase_counts)  # Total actions across all traffic lights

    # Load the trained model
    model = DQN(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load("dqn_traffic_model_episode_100.pth"))
    model.eval()

    state = get_state()

    for step in range(1000):  # Adjust as per simulation duration
        action = choose_action(state, model, action_dim)

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

        # Get next state
        state = get_state()

    traci.close()

if __name__ == "__main__":
    main()