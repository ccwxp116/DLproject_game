import pandas as pd


## download image from url:
data_path = './data.csv'
data = pd.read_csv(data_path)
data.head(), data.columns
screenshot_urls = data['Screenshots']