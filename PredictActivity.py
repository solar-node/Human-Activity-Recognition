import joblib
import pandas as pd
import random
import os

# Define file paths relative to the script's location
DATA_PATH = "Data/test.csv"
# Note: Ensure these model files are in the same root directory as this script
MODEL_PATH = "human_activity_model.pkl" 
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"
ENCODER_PATH = "label_encoder.pkl"

def load_components(model_path, scaler_path, pca_path, encoder_path):
    """Loads all the necessary model components from disk."""
    print("Loading model components...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    label_encoder = joblib.load(encoder_path)
    print("Components loaded successfully.")
    return model, scaler, pca, label_encoder

def predict_random_activity(data_path, model, scaler, pca, label_encoder):
    """
    Loads the test data, selects a random row, and predicts the activity.
    """
    print("\nLoading test data...")
    # We only need the features so we'll drop 'Activity' and 'subject'
    try:
        test_df = pd.read_csv(data_path)
        X_test = test_df.drop(['Activity', 'subject'], axis=1)
    except KeyError:
        # Handle cases where the data might not have the 'subject' column
        test_df = pd.read_csv(data_path)
        X_test = test_df.drop(['Activity'], axis=1)

    # Select a SINGLE RANDOM row for prediction
    num_rows = X_test.shape[0]
    random_index = random.randint(0, num_rows - 1)
    sample_data = X_test.iloc[[random_index]]
    
    print(f"Making a prediction for a random row (index: {random_index}) from test.csv")

    # Preprocess the data (Scale and apply PCA)
    sample_data_scaled = scaler.transform(sample_data)
    sample_data_pca = pca.transform(sample_data_scaled)

    # Make and decode the prediction
    prediction_encoded = model.predict(sample_data_pca)
    prediction = label_encoder.inverse_transform(prediction_encoded)

    # Display the result
    print(f"\nPrediction for row: {random_index} -> The activity is {prediction[0]}\n")

if __name__ == "__main__":
    # This block runs only when the script is executed directly
    try:
        # Load all the components
        model, scaler, pca, le = load_components(MODEL_PATH, SCALER_PATH, PCA_PATH, ENCODER_PATH)
        
        # Run the prediction
        predict_random_activity(DATA_PATH, model, scaler, pca, le)

    except FileNotFoundError as e:
        print(f"\nERROR: A necessary file was not found.")
        print(f"Please make sure the following files exist in the correct locations:")
        print(f"- Dataset at: {DATA_PATH}")
        print(f"- Model files in the root directory (e.g., {MODEL_PATH})")
        print(f"Details: {e}")
