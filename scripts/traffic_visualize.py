import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

data = pd.read_csv("../traffic_data/traffic_data.csv")

# Initialize map centered around Tampere, Finland
m = folium.Map(location=[61.4978, 23.7610], zoom_start=13)

# Prepare data for the heatmap: [[lat, lon, weight]]
heat_data = [[row['Latitude'], row['Longitude'], row['Traffic Count']] for index, row in data.iterrows()]

# Add heatmap layer
HeatMap(heat_data).add_to(m)

# Save map as an HTML file
m.save("traffic_heatmap_morning_8_45.html")