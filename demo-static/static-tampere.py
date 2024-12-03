import traci

def run_simulation():
    # Start the SUMO simulation with the configuration file
    traci.start(["sumo-gui", "-c", "static-tampere.sumocfg"])

    # Run the simulation until the end time (you can modify the end time in the configuration file)
    while traci.simulation.getTime() < 1000:  # Run the simulation for 1000 seconds
        traci.simulationStep()  # Advance the simulation by one step

    # Close the simulation once done
    traci.close()

if __name__ == "__main__":
    run_simulation()