import os
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# Paths
data_folder = r"PROCESSED-DATA/data_processed_combined"
output_folder = r"outputs/data_processed_combined_filtered"

# Define the neural network model architecture
class HiggsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(HiggsClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

# Load the scaler and the model's state dictionary
scaler_path = "scaler.pkl"
model_path = "higgs_classifier_state_dict.pth"

scaler = joblib.load(scaler_path)
input_dim = 24  # Adjust this based on the number of features used in training
model = HiggsClassifier(input_dim=input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Feature columns
features = [
    'Leading_lepton_pt', 'Leading_lepton_eta', 'Leading_lepton_phi', 'Leading_lepton_energy', 'Leading_lepton_PID',
    'Second_lepton_pt', 'Second_lepton_eta', 'Second_lepton_phi', 'Second_lepton_energy', 'Second_lepton_PID',
    'Third_lepton_pt', 'Third_lepton_eta', 'Third_lepton_phi', 'Third_lepton_energy', 'Third_lepton_PID',
    'Fourth_lepton_pt', 'Fourth_lepton_eta', 'Fourth_lepton_phi', 'Fourth_lepton_energy', 'Fourth_lepton_PID',
    'InvMass_Pair1', 'InvMass_Pair2', 'PT_Pair1', 'PT_Pair2'
]

# Function to process and filter each file
def filter_higgs_events(file_path, output_folder):
    # Load data
    df = pd.read_csv(file_path)
    X = df[features].values
    
    # Normalize data
    X = scaler.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Pass data through the model
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
    
    # Select only rows classified as Higgs boson (label = 1)
    higgs_events_df = df[predictions.numpy() == 1]
    
    # Save the filtered data to a new file
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, file_name)
    higgs_events_df.to_csv(output_file_path, index=False)
    
    print(f"Processed {file_name}: {len(higgs_events_df)} Higgs events saved.")

# Loop through all files and apply the filter
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_folder, file_name)
        filter_higgs_events(file_path, output_folder)
