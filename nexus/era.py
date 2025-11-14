import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import socket
import datetime
import pandas as pd
import ee
from config import *
from eradown import dowcon_day

def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def safe_initialize(project_id, max_retries=5, wait_time=10):
    for attempt in range(max_retries):
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            print("Earth Engine initialized successfully.")
            return True
        except Exception as e:
            print(f"Earth Engine initialization failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(wait_time)
    return False


def safe_download(dataset, aoi, date_str, max_retries=20, wait_time=15):
    attempt = 0
    while attempt < max_retries:
        try:
            dowcon_day(dataset, aoi, date_str)
            print(f"Successfully processed {date_str}")
            return True
        except Exception as e:
            print(f"Error processing {date_str} (attempt {attempt+1}/{max_retries}): {e}")
            if not check_internet():
                print("Internet down. Waiting for reconnection...")
                while not check_internet():
                    time.sleep(wait_time)
                print("Internet restored. Retrying download...")
                continue
            else:
                attempt += 1
                time.sleep(wait_time)

    print(f"Skipping {date_str} after repeated failures.")
    return False


if __name__ == "__main__":
    if not safe_initialize(era_project_id):
        print("Could not initialize Earth Engine after several attempts. Exiting.")
        sys.exit(1)

    aoi = ee.Geometry.Polygon(era_coordinates, proj='EPSG:4326', geodesic=False)
    dataset = ee.ImageCollection(era_collection)

    start = datetime.datetime.strptime(era_start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(era_end_date, '%Y-%m-%d')
    delta = datetime.timedelta(days=1)
    current_date = start

    while current_date < end:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Processing {date_str}...")
        safe_download(dataset, aoi, date_str)
        current_date += delta

    print("All dates processed.")
