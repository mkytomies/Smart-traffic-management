import xml.etree.ElementTree as ET

def modify_routes_with_short_vehicle_type(input_file, output_file):
    # Parse the SUMO routes XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Define the new vehicle type
    vType = ET.Element('vType', id="short_passenger_car", accel="2.6", decel="4.5", sigma="0.5", length="3.5", minGap="2.5", maxSpeed="55", guiShape="passenger")
    
    # Add the new vehicle type definition to the root element
    root.insert(0, vType)  # Insert the new vType as the first element
    
    # Iterate through each <flow> element and change its type to the new "short_passenger_car"
    for flow in root.findall('flow'):
        flow.set('type', 'short_passenger_car')

    # Write the modified XML tree back to a new output file
    tree.write(output_file)
    print(f"Routes modified and saved to {output_file}")

# Example usage:
input_file = 'traffic.rou.xml'  # Replace with the actual path to your routes file
output_file = 'modified_traffic.rou.xml'  # Path where you want to save the modified routes file

modify_routes_with_short_vehicle_type(input_file, output_file)
