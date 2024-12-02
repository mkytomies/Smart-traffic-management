import xml.etree.ElementTree as ET

def modify_traffic_light_logic(input_file, output_file):
    # Parse the SUMO network XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Step 1: Set the 'duration' attribute to a minimal value for all <phase> elements inside <tlLogic> elements
    for tl_logic in root.findall('tlLogic'):
        # Iterate through each phase and set the 'duration' attribute to a minimal value
        for phase in tl_logic.findall('phase'):
            if phase.get('duration') is not None:  # Ensure the 'duration' attribute exists
                phase.set('duration', '100000')  # Set a minimal duration value
    
    # Step 2: Save the modified network to a new file
    tree.write(output_file)

    print(f"Traffic light logic modified. Modified file saved as {output_file}")

# Example usage:
input_file = 'tampere.net.xml'  # Replace with your actual network XML file path
output_file = 'modified_tampere.net.xml'  # Path to save the modified file
modify_traffic_light_logic(input_file, output_file)
