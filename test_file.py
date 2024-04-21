import pandas as pd

data = pd.read_csv("/Users/ciciwxp/Desktop/dlproj_data/games.csv")

print(data.shape)

print(data.columns)

selected_columns = ['About the game', 'Reviews', 'Header image', 'Screenshots', 'Genres']

missing_columns = [col for col in selected_columns if col not in data.columns]
if not missing_columns:
    sampled_data = data[selected_columns].sample(n=2000, random_state=42)
else:
    sampled_data = None

(sampled_data.head() if sampled_data is not None else "Missing Columns: " + ", ".join(missing_columns))

sampled_data.shape 
sampled_data.to_csv('/Users/ciciwxp/Desktop/CSCI2470/DLproject_game/data.csv', index=False)