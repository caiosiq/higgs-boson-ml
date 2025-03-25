import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

# Define the model structure
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

# Initialize the model with the correct input dimension
model = HiggsClassifier(input_dim=24)  # Assuming 24 input features

# Load the state dictionary from the saved file
model_path = "higgs_classifier_state_dict.pth"
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode
print("Model state dictionary loaded successfully.")


# Load your data and labels
data_folder = r"C:\Users\CaioV\Dropbox (MIT)\JLab_Higgs_Caio_Marina\Higgs_Boson\MC_processed_combined"

# Function to load data and apply labels based on filename
def load_data_with_labels(folder):
    all_data = []
    all_labels = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder, file_name)
            df = pd.read_csv(file_path)

            # Label: 1 for Higgs files, 0 otherwise
            label = 1 if 'higgs' in file_name.lower() else 0
            all_data.append(df)
            all_labels.extend([label] * len(df))
            
    # Concatenate all data
    data = pd.concat(all_data, ignore_index=True)
    labels = pd.Series(all_labels)
    return data, labels

# Load and preprocess the data
data, labels = load_data_with_labels(data_folder)

# Select features for training
features = [
    'Leading_lepton_pt', 'Leading_lepton_eta', 'Leading_lepton_phi', 'Leading_lepton_energy', 'Leading_lepton_PID',
    'Second_lepton_pt', 'Second_lepton_eta', 'Second_lepton_phi', 'Second_lepton_energy', 'Second_lepton_PID',
    'Third_lepton_pt', 'Third_lepton_eta', 'Third_lepton_phi', 'Third_lepton_energy', 'Third_lepton_PID',
    'Fourth_lepton_pt', 'Fourth_lepton_eta', 'Fourth_lepton_phi', 'Fourth_lepton_energy', 'Fourth_lepton_PID',
    'InvMass_Pair1', 'InvMass_Pair2', 'PT_Pair1', 'PT_Pair2'
]
X = data[features].values
y = labels.values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Make predictions
with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted_labels = torch.max(outputs, 1)

# Calculate initial Higgs fraction
initial_higgs_fraction = np.sum(y) / len(y)

# Calculate post-model Higgs fraction (precision of the model on Higgs class)
post_model_higgs_fraction = precision_score(y, predicted_labels.numpy(), pos_label=1)

# Calculate signal improvement
signal_improvement = post_model_higgs_fraction / initial_higgs_fraction if initial_higgs_fraction > 0 else float('inf')

# Print results
print(f"Initial Higgs Fraction: {initial_higgs_fraction:.4f}")
print(f"Post-Model Higgs Fraction (Precision): {post_model_higgs_fraction:.4f}")
print(f"Signal Improvement Factor: {signal_improvement:.2f}")
