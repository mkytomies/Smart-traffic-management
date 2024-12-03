import traci
import traci.constants as tc
import os

def start_sumo_simulation():
    """Starts the SUMO simulation with the specified configuration file."""
    sumo_binary = "sumo"  # Use "sumo-gui" for graphical interface
    config_file = "simulation.sumocfg"

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    traci.start([sumo_binary, "-c", config_file])

def fetch_vehicle_data():
    """Fetches and prints data for all vehicles in the simulation."""
    vehicles = traci.vehicle.getIDList()
    print(f"Number of vehicles: {len(vehicles)}")
    
    for vehicle_id in vehicles:
        position = traci.vehicle.getPosition(vehicle_id)
        speed = traci.vehicle.getSpeed(vehicle_id)
        print(f"Vehicle ID: {vehicle_id}, Position: {position}, Speed: {speed} m/s")

def fetch_traffic_light_data():
    """Fetches and prints data for all traffic lights in the simulation."""
    traffic_lights = traci.trafficlight.getIDList()
    print(f"Number of traffic lights: {len(traffic_lights)}")

    for tl_id in traffic_lights:
        state = traci.trafficlight.getRedYellowGreenState(tl_id)
        print(f"Traffic Light ID: {tl_id}, Current State: {state}")

def edit_traffic_light(tl_id, new_program):
    """Edits the program of a specified traffic light."""
    if tl_id not in traci.trafficlight.getIDList():
        print(f"Traffic light '{tl_id}' does not exist.")
        return

    traci.trafficlight.setRedYellowGreenState(tl_id, new_program)
    print(f"Traffic Light ID: {tl_id} updated to new program: {new_program}")

def main():
    try:
        start_sumo_simulation()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Fetch and display vehicle data
            fetch_vehicle_data()

            # Fetch and display traffic light data
            fetch_traffic_light_data()

            # Example: Automatically change traffic light states (for demonstration purposes)
            traffic_lights = traci.trafficlight.getIDList()
            for tl_id in traffic_lights:
                # Get the current program and modify it (this is an example of a toggle)
                current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                new_state = current_state[::-1]  # Reverse state as a simple modification
                edit_traffic_light(tl_id, new_state)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        traci.close()

if __name__ == "__main__":
    main()

