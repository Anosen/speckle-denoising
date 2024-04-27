import os
from pathlib import Path

from config import DATA_DIR, PROJECT_DIR

def download_data(token_path: str):
    # Check if the data is not already downloaded
    if os.path.exists(DATA_DIR):
        print("Data already downloaded!")
        return
    
    try:
        #make sure kaggle.json file is present
        os.system(f'ls -lha {token_path}')
        
        # Assert that it is named kaggle.json
        assert 'kaggle.json' in token_path, 'Token file should be named kaggle.json'

        #Install kaggle API client
        os.system('pip install -q kaggle')

        # Move the kaggle.json token to ~/.kaggle, where the Kaggle API client expects to find it. 
        os.system(f'mkdir -p {PROJECT_DIR}/.kaggle')
        os.system(f'cp {token_path} ~/.kaggle/')

        # Set permissions
        os.system(f'chmod 600 {PROJECT_DIR}/.kaggle/kaggle.json')

        # Check the current directory before downloading the datasets
        os.system('pwd')

        # List all available datasets
        os.system('kaggle competitions list')
        
        # Create the data directory
        os.system(f'mkdir -p {DATA_DIR}')
        
        # Move to the data directory
        os.chdir(DATA_DIR)

        #download the required dataset from kaggle
        os.system('kaggle competitions download -c ultrasound-nerve-segmentation')

        #If your file is a zip file you can unzip with the following code
        os.system('unzip ultrasound-nerve-segmentation.zip')
        
    except Exception as e:
        print(f"An error occured: {e}")
        import traceback; traceback.print_exc();
    
    print("Data downloaded successfully!")
    return