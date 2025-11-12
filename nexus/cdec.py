import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import requests
import pandas as pd
from io import StringIO
import os

os.makedirs("../dataset", exist_ok=True)
response = requests.get(base_url, params=params)
if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text))
    df.to_csv(save_path, index=False)
    print("Data saved to " + save_path)
else:
    print(f"Error: {response.status_code} - {response.text}")
