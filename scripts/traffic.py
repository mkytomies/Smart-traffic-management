import requests
import pandas as pd
from datetime import datetime, timedelta

# Define your devices with names and coordinates
devices = {
    "tre327" : {"lat": 61.501577, "lon": 23.770451},
    "tre227" : {"lat": 61.498572, "lon": 23.771758},
    "tre158" : {"lat": 61.497491, "lon": 23.772112},
    "tre112" : {"lat": 61.495771, "lon": 23.772630},
    "tre106" : {"lat": 61.495325, "lon": 23.769085},
    "tre115" : {"lat": 61.494865, "lon": 23.772034},
    "tre402" : {"lat": 61.498975, "lon": 23.779556},
    "tre412" : {"lat": 61.501050, "lon": 23.786898},
    "tre401" : {"lat": 61.499068, "lon": 23.787050},
    "tre428" : {"lat": 61.498942, "lon": 23.784002},
    "tre427" : {"lat": 61.498287, "lon": 23.787051},
    "tre425" : {"lat": 61.498271, "lon": 23.792154},
    "tre545" : {"lat": 61.495070, "lon": 23.794690},
    "tre123" : {"lat": 61.495199, "lon": 23.790101},
    "tre125" : {"lat": 61.495275, "lon": 23.783948},
}

# Define the base URL of the API
api_base_url = "http://trafficlights.tampere.fi/api/v1/trafficAmount/"

# Set the time range for the API call
start_time = "2024-11-12T07:00"
end_time = "2024-11-12T07:15"

# Initialize an empty list to store the data
all_data = []

# Loop through each device and make API calls
for device_name, coords in devices.items():
    url = f"{api_base_url}{device_name}?startTime={start_time}&endTime={end_time}"
    
    try:
        # Make the API call
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'})
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json().get("results", [])
        
        # Process each entry in the response
        for entry in data:
            # Extract relevant fields
            timestamp = entry["tsPeriodEnd"]
            traffic_count = entry["trafficAmount"] if entry["trafficAmount"] is not None else 0
            
            # Append to the all_data list with additional details
            all_data.append({
                "Device Name": device_name,
                "Latitude": coords["lat"],
                "Longitude": coords["lon"],
                "Timestamp": timestamp,
                "Traffic Count": traffic_count
            })
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {device_name}: {e}")

# Convert to a DataFrame for easy manipulation
df = pd.DataFrame(all_data)

# Convert the timestamp to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort by Timestamp if needed
df = df.sort_values(by="Timestamp")

# Save to CSV or another format
df.to_csv("traffic_data_7.csv", index=False)

# Calculate the total traffic count by detector
total_traffic_by_detector = df.groupby("Device Name")["Traffic Count"].sum()

# Print the total traffic count by detector
print("Total traffic 7:00-7:15 count by detector:")
print(total_traffic_by_detector)

print("Data fetched and organized successfully.")
