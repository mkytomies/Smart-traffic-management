import sumolib
import xml.etree.ElementTree as ET

net = sumolib.net.readNet('map_no_uturns.net.xml')

tree = ET.parse('map_no_uturns.net.xml')
root = tree.getroot()

special_char_map = {ord('ä'):'a', ord('ö'):'o', ord('å'):'o'}

# Check the first edge to see available methods
first_edge = net.getEdges()[0]
print(dir(first_edge))

# Dictionary to track the count of edges for each road
road_counter = {}

id_mapping = {}
edge_mapping = {}

# Open the network file for modification
with open('map_renamed.net.xml', 'w') as f:
    # Iterate through the edges
    for edge in net.getEdges():
         
        road_name = edge.getName()
        
        # Check if the road has been encountered before and update the counter
        if road_name in road_counter:
            road_counter[road_name] += 1
        else:
             road_counter[road_name] = 1

        new_road_name = road_name.translate(special_char_map)
        new_road_name = new_road_name.replace(' ', '_')
        
        # Create a unique edge ID by appending the counter to the road name
        unique_edge_id = f"{new_road_name}#{road_counter[road_name]}"

        old_edge_id = edge.getID()

        id_mapping[old_edge_id] = unique_edge_id
         
        xml_edge = root.find(f".//edge[@id='{edge.getID()}']")

        if xml_edge is not None :
            xml_edge.set('id', unique_edge_id)
        else :
            print("No edge found in XML")

    for connection in root.findall(".//connection") :
        from_edge = connection.get('from')
        to_edge = connection.get('to')

        if from_edge in id_mapping :
            connection.set('from', id_mapping[from_edge])
        if to_edge in id_mapping :
            connection.set('to', id_mapping[to_edge])

    for roundbout in root.findall(".//roundabout") :
        edges = roundbout.get('edges')
        edges_list = edges.split()
        new_edges_list = []

        for edge_id in edges_list :
            if edge_id in id_mapping :
                new_edge_id = id_mapping[edge_id]
                new_edges_list.append(new_edge_id)
            else :
                print(f"Edge {edge_id} not found")

        roundbout.set('edges', ' '.join(new_edges_list))

        


tree.write('renamed.net.xml')

print("Edge renaming completed with unique IDs.")
