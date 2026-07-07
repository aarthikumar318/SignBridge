import pandas as pd

old = pd.read_csv("data.csv", header=None)
new = pd.read_csv("data_new.csv", header=None)

combined = pd.concat([old, new], ignore_index=True)
combined.to_csv("final_data.csv", index=False, header=False)

print("Merged successfully!")