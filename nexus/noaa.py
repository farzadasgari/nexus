import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

STATION_ID = noaa_station
DATASET = noaa_dataset
DATATYPES = noaa_datatypes
START_DATE = datetime.strptime(noaa_start_date, '%Y-%m-%d')
END_DATE = datetime.strptime(noaa_end_date, '%Y-%m-%d')

TOKEN = noaa_token

headers = {"token": TOKEN}

all_data = []

current_start = START_DATE
while current_start < END_DATE:
    current_end = min(current_start + timedelta(days=364), END_DATE)
    start_str = current_start.strftime('%Y-%m-%d')
    end_str = current_end.strftime('%Y-%m-%d')
    print(f"Processing {start_str} to {end_str}...")

    for dtype in DATATYPES:
        print(f"  Fetching {dtype}...")
        url = noaa_base_url
        params = {
            "datasetid": DATASET,
            "stationid": STATION_ID,
            "datatypeid": dtype,
            "startdate": start_str,
            "enddate": end_str,
            "limit": 1000,
            "units": "metric",
            "includemetadata": "false"
        }

        offset = 1
        while True:
            params["offset"] = offset
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"    Error {dtype}: {response.status_code} - {response.text}")
                break
            data = response.json()
            results = data.get("results", [])
            if not results:
                break
            all_data.extend(results)
            offset += len(results)
            print(f"    → {offset-1} records so far...")

    current_start = current_end + timedelta(days=1)
    time.sleep(1)

if all_data:
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.pivot(index='date', columns='datatype', values='value').reset_index()
    df.columns.name = None
    os.makedirs("../dataset", exist_ok=True)
    df.to_csv(noaa_save_path, index=False)
    print("NOAA Data Saved.")
else:
    print("No data fetched. Check token/station.")
