import pandas as pd
import re

data = pd.read_csv("/Users/ciciwxp/Desktop/dlproj_data/games.csv")

print(data.shape)

print(data.columns)

selected_columns = ['About the game', 'Header image', 'Screenshots', 'Genres']
data = data[selected_columns].dropna()

#english only
english_pattern = re.compile(r'^[A-Za-z0-9\s\.,;:"\'?!-]+$')
def is_english_and_long(text):
    return bool(english_pattern.match(text)) and len(text) >= 30
data = data[data['About the game'].apply(is_english_and_long)]

data = data[data['Genres'].str.contains('Adventure|Casual|Strategy|Action|Simulation', na=False, case=False)]

print(data.shape)

desired_genres = ['Adventure', 'Casual', 'Strategy', 'Action', 'Simulation']

def find_first_matching_genre(genres_string):
    genres_list = genres_string.split(',')
    for genre in genres_list:
        if genre.strip() in desired_genres:
            return genre.strip()
    return None  # In case no desired genres are found

# Apply the function to each row in the 'Genres' column
data['Genres'] = data['Genres'].apply(find_first_matching_genre)

# Display the updated DataFrame
data['Genres'].value_counts()
print(data.head())


# data_expanded['Genres'] = data['Genres'].str.split(',')
# data_expanded = data['Genres'].apply(pd.Series)
# data_expanded.columns = ['Genre{}'.format(i + 1) for i in range(data_expanded.shape[1])]
# data_expanded = pd.concat([data.drop('Genres', axis=1), data_expanded], axis=1)
# print(data_expanded.head(60))
# data_expanded['Genre1'].value_counts()


# Combine the conditions across all genre columns
sampled_data = pd.DataFrame()  # Initialize an empty DataFrame to store sampled data

for genre in desired_genres:
    # Filter data for the current genre
    genre_data = data[data['Genres'] == genre]
    
    # Check if there are at least 1000 rows, if not, take all available rows
    num_samples = min(len(genre_data), 1000)
    
    # Randomly sample 1000 rows (or less if not available)
    genre_sample = genre_data.sample(n=num_samples, random_state=1)  # Random state for reproducibility
    
    # Append the sampled data to the overall DataFrame
    sampled_data = pd.concat([sampled_data, genre_sample], ignore_index=True)

# Display the shape to confirm it has 5000 rows (or less if some genres didn't have enough data)
print(sampled_data.shape)
# Display the first few rows of the sampled data
print(sampled_data.head())


sampled_data.shape 
sampled_data.to_csv('/Users/ciciwxp/Desktop/CSCI2470/DLproject_game/data.csv', index=False)