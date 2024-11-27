import pandas as pd

df = pd.read_csv("../traffic_data/traffic_count_intervals.csv")

max_traffic_count = df["Traffic Count"].max()
#print(f"Maximum traffic count recorded: {max_traffic_count}")

traffic_count = df["Traffic Count"]
traffic_count = round((traffic_count / max_traffic_count) * 500, 0)
print(traffic_count)

df["Traffic Count"] = traffic_count
print(df)

df.to_csv("normalized_traffic_count_intervals.csv", index=False)