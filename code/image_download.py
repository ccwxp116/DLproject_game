import multiprocessing
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm


## download image from url:

# path:
data_path = 'data/data_genre.csv' #change to test or train
image_folder = Path('data/image/genre_screenshots') #change to test or train
image_folder.mkdir(parents=True, exist_ok=True)

# code:
data = pd.read_csv(data_path)
first_urls = data['Screenshots'].apply(lambda x: x.split(',')[0]) #if there is multiple, only download the first url

'''
def download_and_name_by_index(url, dest_folder, index, timeout=10):
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status() 

        image_path = dest_folder / f"{index}.jpg"
        with open(image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return image_path
    except requests.RequestException as e:
        print(f"Request error for {url}: {e}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return None




# Download images with new naming convention and format
indexed_downloads = [download_and_name_by_index(url, image_folder, idx) 
                     for idx, url in tqdm(enumerate(first_urls))]
# Output the result of these downloads
#indexed_downloads
'''

def download_and_name_by_index(args):
    url, dest_folder, index = args
    timeout = 10
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        image_path = dest_folder / f"{index}.jpg"
        with open(image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return image_path
    except requests.RequestException as e:
        print(f"Request error for {url}: {e}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return None

def main():
    # Prepare arguments for the download function
    args_list = [(url, image_folder, idx) for idx, url in enumerate(first_urls)]
    
    # Set up multiprocessing pool
    with multiprocessing.Pool(processes=30) as pool:
        # Map the download function to the arguments
        results = list(tqdm(pool.imap(download_and_name_by_index, args_list), total=len(args_list)))

    # Output the result of these downloads
    print(results)

if __name__ == "__main__":
    main()