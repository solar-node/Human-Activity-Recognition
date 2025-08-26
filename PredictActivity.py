import joblib
import pandas as pd
import random
# Load the saved model
model = joblib.load("har-api/HAR-model.pkl")
scaler = joblib.load("har-api/scaler.pkl")
pca = joblib.load("har-api/pca.pkl")
label_encoder = joblib.load("har-api/label_encoder.pkl")


# We only need the features so we'll drop 'Activity and 'subject'

try:
    test_df = pd.read_csv("test.csv")
    X_test = test_df.drop(['Activity', 'subject'], axis =1 )
except KeyError:
    # Handle the subjects where data might not have the 'subject'
    test_df = pd.read_csv("test.csv")
    X_test = test_df.drop(['Activity'], axis = 1)

# Select a SINGLE RANDOM row for prediction
# Get the total number of rows in the test data
num_rows = X_test.shape[0]

# Generate a random index number
random_index = random.randint(0, num_rows - 1)

sample_data = X_test.iloc[[random_index]]
print(f"Making a prediction for a random row (index: {random_index}) from test.csv")

# Scale the data
sample_data_scaled = scaler.transform(sample_data)

# Apply PCA
sample_data_pca = pca.transform(sample_data_scaled)

# Prediction using model (encoded)
prediction_encoded = model.predict(sample_data_pca)

# Decode the prediction to get the activity name
prediction = label_encoder.inverse_transform(prediction_encoded)

# Display the result
print(f"\nPrediction for row: {random_index} -> The activity is {prediction[0]}\n")

