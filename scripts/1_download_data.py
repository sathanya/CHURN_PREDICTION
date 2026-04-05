import os
import requests

def download_uci_dataset(download_path="data/raw"):
    os.makedirs(download_path, exist_ok=True)
    
    # Official URL for the UCI Online Retail dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    file_path = os.path.join(download_path, "online_retail.xlsx")
    
    print(f"Downloading UCI Online Retail dataset to {file_path}...")
    print("This file is ~23MB. Please wait...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print("Download complete!")
                
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    download_uci_dataset()
