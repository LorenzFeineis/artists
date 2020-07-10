import pandas as pd

artists_info = pd.read_csv("../artists.csv")
artists_info["name"][19] = "Albrecht Duerer"
print(artists_info["name"][19])
artists_info.to_csv("../artists_changed.csv")

print("Changed family of Albrecht to Duerer.")
