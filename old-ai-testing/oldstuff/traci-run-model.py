import torch
import torch.nn as nn 
import traci
import numpy as np

# DQN Class Definition (same as used during training)
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

# Load trained model
def load_model(state_dim, action_dim, model_path):
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Choose best action
def choose_action(state, model):
    state_tensor = torch.tensor([state], dtype=torch.float32)
    with torch.no_grad():
        q_values = model(state_tensor)
    return q_values.argmax().item()

def connect_to_sumo(sumo_binary="sumo", config_file="./simulation.sumocfg"):
    sumo = [
    "sumo",
    "-c", "simulation.sumocfg",
    "--collision.action", "none",  # No teleporting on collisions
    "--time-to-teleport", "-1",  # Disable teleporting for blocked vehicles
    "--no-step-log",  # Suppress log spam
    "--no-warnings"  # Suppress warnings
    ]
    traci.start(sumo)

# Main simulation loop
def run_sumo_with_model(model, state_dim, action_dim):
    connect_to_sumo()
    traci.load(["-c", "simulation.sumocfg"])
    traffic_lights = traci.trafficlight.getIDList()
    phase_counts = get_traffic_light_phase_counts()

    state = get_state()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Choose an action
        action = choose_action(state, model)

        # Map action to traffic light phases
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
    # Parameters (must match training)
    state_dim = 60  # FIXED_STATE_DIM
    action_dim = 24  # Replace with the number of actions from training
    model_path = "dqn_traffic_model_episode_400.pth"

    # Load the trained model
    model = load_model(state_dim, action_dim, model_path)

    # Run the simulation with the trained model
    run_sumo_with_model(model, state_dim, action_dim)