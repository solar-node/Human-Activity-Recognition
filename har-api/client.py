import pandas as pd
import json
import requests # Import the requests library
import random

# --- 1. Prepare the Data (same as before) ---
# Load your test data
test_df = pd.read_csv('test.csv')
X_test = test_df.drop(['Activity', 'subject'], axis=1)

# Get a random row from the test data
random_index = random.randint(0, len(X_test) - 1)
row_features = X_test.iloc[random_index].tolist()

# Create the JSON object your API expects
api_request_data = {"features": row_features}


# --- 2. Send the Data to the API ---
# The URL of your running FastAPI server's prediction endpoint
api_url = "http://127.0.0.1:8000/predict"

print(f"Sending data from row {random_index} to the API...")

try:
    # Make the POST request
    response = requests.post(api_url, json=api_request_data)

    # Raise an exception if the request was unsuccessful (e.g., 404, 500)
    response.raise_for_status() 

    # --- 3. Print the Response from the API ---
    prediction = response.json() # Convert the JSON response to a Python dictionary
    print("\n--- Prediction Received from API ---")
    print(prediction)

except requests.exceptions.RequestException as e:
    print(f"\n--- An error occurred ---")
    print(f"Could not connect to the API. Is the server running?")
    print(f"Error details: {e}")

