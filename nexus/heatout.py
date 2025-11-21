import pandas as pd

heatwaves = pd.read_csv("../dataset/heatwaves.csv", parse_dates=["Date"])
droughts  = pd.read_csv("../dataset/droughts.csv",  parse_dates=["Date"])
cols_to_drop = ["PRCP", "TMAX", "TMIN"]
droughts = droughts.drop(columns=cols_to_drop, errors="ignore")
heatout = heatwaves.merge(droughts, on="Date", how="left")
heatout.to_csv("../dataset/heatout.csv", index=False)
