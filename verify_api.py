import pandas as pd
import requests
import json

# Load a sample from test.csv
test_df = pd.read_csv('data/processed/test.csv')
# Get features in correct order (all except Class)
features = [col for col in test_df.columns if col != 'Class']
sample = test_df[features].iloc[0:1].to_dict(orient='records')

# API URL
url = "http://localhost:8000/predict"

try:
    response = requests.post(url, json=sample)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
