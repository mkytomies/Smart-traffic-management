import traci
from traci.step import StepListener

simulation = "../test.sumocfg"
# "../old-ai-testing/simulation.sumocfg"


def get_all_traffic_lights():
    """
    Fetches the IDs of all traffic lights in the simulation.
    
    Returns:
        list: A list of traffic light IDs.
    """
    tls_ids = traci.trafficlight.getIDList()
    return tls_ids

def get_vehicle_data(lane_id):
    # Get the IDs of the vehicles currently in this lane
    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
    
    # Calculate the number of vehicles (vehicle count)
    vehicle_count = len(vehicles)
    
    # Get the number of vehicles currently halting in the lane (queue length)
    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
    
    # Calculate the average wait time of the vehicles in the lane
    avg_wait_time = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicles) / max(len(vehicles), 1)

    # Return all relevant data, including vehicle_count
    return {
        "lane_id": lane_id,
        "vehicle_count": vehicle_count,  # Add vehicle_count here
        "queue_length": queue_length,
        "avg_wait_time": avg_wait_time
    }

def get_tls_data(tls_id):
    
        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        last_switch_time = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
        elapsed_time = abs(last_switch_time)  # Ensure non-negative
        
        lane_metrics = [get_vehicle_data(lane) for lane in set(controlled_lanes)]
        
        return {
            "tls_id": tls_id,
            "current_state": current_state,
            "elapsed_time": elapsed_time,
            "lane_metrics": lane_metrics
        }

def switch_traffic_light(light_id, target_color):
    """
    Switches the specified traffic light to the target color and ensures conflicting traffic lights are red.

    Parameters:
        light_id (str): The ID of the traffic light to control.
        target_color (str): The target color for the traffic light ("green" or "red").
    """
    # Validate the input color
    if target_color not in ["red", "green"]:
        raise ValueError(f"Invalid target color: {target_color}. Must be 'red' or 'green'.")

    # Get the current traffic light state and controlled lanes
    current_state = traci.trafficlight.getRedYellowGreenState(light_id)
    controlled_lanes = traci.trafficlight.getControlledLanes(light_id)

    # Get conflicting lights in the same junction
    conflicting_lights = get_conflicting_lights(light_id)

    # Set conflicting lights to red
    for conflicting_light in conflicting_lights:
        # Get the current state of the conflicting traffic light
        conflicting_state = traci.trafficlight.getRedYellowGreenState(conflicting_light)
        
        # Build the red state string with the same number of 'r' as phases in the conflicting light's state
        red_state = 'r' * len(conflicting_state)
        
        # Set all phases of the conflicting traffic light to red
        traci.trafficlight.setRedYellowGreenState(conflicting_light, red_state)
        
        print(f"Conflicting traffic light {conflicting_light} switched to red.")

    # Prepare the new traffic light state (We are working with a list to modify it)
    new_state = list(current_state)

    # Check for yellow-red phases and ensure they are skipped, jumping to green
    for idx, phase_state in enumerate(current_state):
        # If the phase contains red or yellow, we should skip it and go to green
        if "r" in phase_state or "y" in phase_state or "R" in phase_state or "Y" in phase_state:
            # If it's red or yellow, set this phase to green or skip it depending on the target color
            new_state[idx] = "g" if target_color == "green" else "r"
        else:
            # Otherwise, retain the current state (for green phases)
            new_state[idx] = phase_state  # Keep green phases as is

    # Apply the new traffic light state
    traci.trafficlight.setRedYellowGreenState(light_id, "".join(new_state))
    print(f"Traffic light {light_id} switched to {target_color}.")







def get_conflicting_lights(tls_id):
    """
    Get all traffic lights in the same junction as the given traffic light ID.

    Parameters:
        tls_id (str): The ID of the traffic light that is about to change.

    Returns:
        list: A list of traffic light IDs in the same junction, excluding the current one.
    """
    # Fetch all traffic light IDs in the network
    tls_ids = traci.trafficlight.getIDList()

    # Get the controlled lanes of the current traffic light
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)

    # Extract junction ID from one of the controlled lanes (assuming all lanes in this light belong to the same junction)
    junction_id = controlled_lanes[0].split("_")[0]  # Extracting the junction ID from the lane

    # Create a list to store conflicting traffic light IDs
    conflicting_tls = []

    # Iterate through all traffic lights to find those in the same junction
    for tls in tls_ids:
        if tls == tls_id:  # Skip the current traffic light
            continue

        # Get controlled lanes for each traffic light
        other_controlled_lanes = traci.trafficlight.getControlledLanes(tls)

        # If any of the lanes from the other traffic light belong to the same junction, it's a conflicting light
        for lane in other_controlled_lanes:
            if lane.split("_")[0] == junction_id:  # Compare junction ID of the lane
                conflicting_tls.append(tls)
                break  # No need to check other lanes for this traffic light

    return conflicting_tls

def list_junctions_with_stoplights():
    """
    Lists all junctions that have stoplights and prints how many stoplights are in each junction.
    """
    # Fetch all traffic lights (stoplights) in the network
    tls_ids = traci.trafficlight.getIDList()

    # Create a dictionary to store junctions and the count of stoplights at each
    junction_stoplight_count = {}

    # Iterate over each traffic light and determine which junction it controls
    for tls_id in tls_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        
        # For each controlled lane, extract the junction ID (by looking at the lane prefix)
        for lane in controlled_lanes:
            # Extract the junction ID from the lane
            junction_id = lane.split("_")[0]  # Assuming the lane ID starts with the junction ID
            if junction_id not in junction_stoplight_count:
                junction_stoplight_count[junction_id] = 1
            else:
                junction_stoplight_count[junction_id] += 1

    # Print the junctions with their stoplight counts
    for junction_id, count in junction_stoplight_count.items():
        print(f"Junction {junction_id} has {count} stoplight(s).")

class MyStepListener(StepListener):
    def step(self, step):
        
        traffic_lights = get_all_traffic_lights()
        
        for tls_id in traffic_lights:
            tls_data = get_tls_data(tls_id)
            
            # Check if lane metrics exist
            if tls_data["lane_metrics"]:
                for lane_data in tls_data["lane_metrics"]:
                    # If vehicle count exceeds threshold, consider switching to green
                    if lane_data["vehicle_count"] > 1:
                        # Get current state of the traffic light
                        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                        
                        # Get the current active phase index
                        current_phase_index = traci.trafficlight.getPhase(tls_id)

                        # Check the state of the current phase
                        phase_state = current_state[current_phase_index]

                        # If the current phase doesn't have green, switch to green
                        if "G" not in phase_state and "g" not in phase_state:
                            switch_traffic_light(tls_id, "green")
        
        return True  # Returning True allows the simulation to continue



def set_all_traffic_lights_red():
    # Get the list of all traffic light IDs
    traffic_light_ids = traci.trafficlight.getIDList()
    print(f"Found {len(traffic_light_ids)} traffic lights.")
    
    # Set each traffic light to red
    for tl_id in traffic_light_ids:
        # Get the number of phases by checking the current state (this assumes all phases have the same length)
        current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
        
        # Build the red state string with the same number of 'r' as phases in the current state
        red_state = 'r' * len(current_state)
        
        # Set all phases to red
        traci.trafficlight.setRedYellowGreenState(tl_id, red_state)
    
    print("All traffic lights set to red.")


# Flag to check if the function has already been executed
has_executed = False

def run_once():
    global has_executed
    if not has_executed:
        set_all_traffic_lights_red()
        list_junctions_with_stoplights()
        has_executed = True  # Set the flag to True after execution


def simulation_loop():
    traci.start(["sumo-gui", "-c", simulation])
    # Run the function only once
    run_once()
    traci.addStepListener(MyStepListener())  # Add the step listener
    while traci.simulation.getMinExpectedNumber() > 0:  # Run while vehicles are in the simulation
        traci.simulationStep()  # Listener automatically processes each step
    traci.close()

if __name__ == "__main__":
    simulation_loop()