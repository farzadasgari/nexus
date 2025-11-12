import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import requests
import pandas as pd
from io import StringIO

os.makedirs("../dataset", exist_ok=True)
response = requests.get(cdec_base_url, params=cdec_params)
if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text))
    df.to_csv(cdec_save_path, index=False)
    print("CDEC Data Saved.")
else:
    print(f"Error: {response.status_code} - {response.text}")
