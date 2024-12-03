import traci
import time

# Configuration

GREEN_DURATION = 20.0  # Time in seconds for green light
RED_DURATION = 20.0    # Time in seconds for red light
YELLOW_DURATION = 3.0  # Fixed duration for yellow light in seconds
MIN_GREEN_RED_DURATION = 15.0  # Minimum green/red light duration

def get_vehicle_data(lane_id):
    """
    Collects vehicle data for a specific lane.

    Args:
        lane_id (str): ID of the lane.

    Returns:
        dict: Lane-specific metrics, including queue length and average wait time.
    """
    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
    avg_wait_time = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicles) / max(len(vehicles), 1)
    
    return {
        "lane_id": lane_id,
        "queue_length": queue_length,
        "avg_wait_time": avg_wait_time
    }

def get_tls_data(tls_id):
    """
    Collects traffic light and lane data for AI decision-making.

    Args:
        tls_id (str): Traffic light system ID.

    Returns:
        dict: Traffic light metrics, including current state, elapsed time, and lane metrics.
    """
    current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    last_switch_time = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
    elapsed_time = abs(last_switch_time)  # Ensure non-negative
    
    # Gather lane data
    lane_metrics = [get_vehicle_data(lane) for lane in set(controlled_lanes)]
    
    return {
        "tls_id": tls_id,
        "current_state": current_state,
        "elapsed_time": elapsed_time,
        "lane_metrics": lane_metrics
    }

def switch_light_periodically(tls_id, current_time, last_switch_time, current_state):
    """
    Periodically switches traffic lights between green and red with a fixed yellow phase.

    Args:
        tls_id (str): Traffic light system ID.
        current_time (float): Current simulation time.
        last_switch_time (float): Last time the light was switched.
        current_state (str): Current state of the traffic light.

    Returns:
        str: Updated state of the traffic light.
    """
    time_elapsed = current_time - last_switch_time

    if "g" in current_state and time_elapsed >= GREEN_DURATION:
        # Switch to yellow before red
        traci.trafficlight.setRedYellowGreenState(tls_id, "y" * len(current_state))
        time.sleep(YELLOW_DURATION)  # Wait for yellow duration
        traci.trafficlight.setRedYellowGreenState(tls_id, "r" * len(current_state))
        return "r" * len(current_state), current_time

    elif "r" in current_state and time_elapsed >= RED_DURATION:
        # Switch to green
        traci.trafficlight.setRedYellowGreenState(tls_id, "g" * len(current_state))
        return "g" * len(current_state), current_time

    # No state change
    return current_state, last_switch_time

# Example simulation loop
def simulation_loop():
    """
    Main simulation loop for periodic traffic light control.
    """
    traci.start(["sumo-gui", "-c", "simulation.sumocfg"])  # Start SUMO with your configuration file
    last_switch_times = {}  # Store last switch times for each traffic light
    states = {}  # Track current state of each traffic light

    try:
        while traci.simulation.getMinExpectedNumber() > 0:  # Run while vehicles are in the simulation
            current_time = traci.simulation.getTime()

            for tls_id in traci.trafficlight.getIDList():
                # Initialize last switch time and state if not set
                if tls_id not in last_switch_times:
                    last_switch_times[tls_id] = 0
                    states[tls_id] = "r" * len(traci.trafficlight.getRedYellowGreenState(tls_id))

                # Perform periodic switch
                current_state = states[tls_id]
                last_switch_time = last_switch_times[tls_id]
                new_state, new_last_switch_time = switch_light_periodically(
                    tls_id, current_time, last_switch_time, current_state
                )

                # Update state and last switch time
                states[tls_id] = new_state
                last_switch_times[tls_id] = new_last_switch_time

            traci.simulationStep()  # Advance the simulation step
    finally:
        traci.close()
if __name__ == "__main__":
    simulation_loop()
