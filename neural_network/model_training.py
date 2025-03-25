import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Add the path to include scaling function
sys.path.append(r"src")
from get_scalling_funcs import get_scale_factor  # Import scaling function

# Define paths
data_folder = r"PROCESSED-DATA/MC_processed_combined"

# Function to load data, apply labels, and scale based on file importance
def load_data_with_labels_and_scaling(folder):
    all_data = []
    all_labels = []
    scales = []
    
    for file_name in os.listdir(folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder, file_name)
            df = pd.read_csv(file_path)

            # Label: 1 for Higgs files, 0 otherwise
            label = 1 if 'higgs' in file_name.lower() else 0
            scale_factor = get_scale_factor(file_name)
            
            scales.extend([scale_factor] * len(df))
            all_data.append(df)
            all_labels.extend([label] * len(df))
            
    # Concatenate all data
    data = pd.concat(all_data, ignore_index=True)
    labels = pd.Series(all_labels)
    return data, labels, scales

# Load data with scaling and labels
data, labels, scales = load_data_with_labels_and_scaling(data_folder)

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
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Split into training and validation sets
X_train, X_val, y_train, y_val, scales_train, scales_val = train_test_split(
    X, y, scales, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
scales_train_tensor = torch.tensor(scales_train, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, scales_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor, torch.tensor(scales_val, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define the neural network model
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

# Initialize model
model = HiggsClassifier(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Custom significance-based training loop
def train_model(model, train_loader, val_loader, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        S_total, B_total = 0.0, 0.0  # For significance calculation

        for X_batch, y_batch, scale_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate true positives (S) and false positives (B) with scales
            S = torch.sum(scale_batch * (predictions == 1) * (y_batch == 1))
            B = torch.sum(scale_batch * (predictions == 1) * (y_batch == 0))
            
            # Ensure S and B require gradients
            S = S.clone().detach().requires_grad_(True)
            B = B.clone().detach().requires_grad_(True)
            
            # Calculate significance loss and backward pass
            significance_loss = -(S / torch.sqrt(S + B + 1e-6))  # Add small epsilon to prevent division by zero
            significance_loss.backward()
            optimizer.step()
            
            train_loss += significance_loss.item()
            S_total += S.item()
            B_total += B.item()

        # Epoch-level significance
        significance = S_total / np.sqrt(S_total + B_total) if (S_total + B_total) > 0 else 0.0

        # Validation phase
        model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            for X_batch, y_batch, _ in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        # Calculate validation metrics
        precision = precision_score(all_labels, all_preds, pos_label=1)
        recall = recall_score(all_labels, all_preds, pos_label=1)
        f1 = f1_score(all_labels, all_preds, pos_label=1)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Significance: {significance:.4f}, Accuracy: {accuracy:.2f}, "
              f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Train the model
train_model(model, train_loader, val_loader, optimizer, epochs=20)

# Save the model and scaler after training
torch.save(model.state_dict(), "higgs_classifier_state_dict.pth")
print("Model state dictionary saved.")

# Save the scaler
import joblib
scaler_path = "scaler.pkl"
joblib.dump(scaler, scaler_path)
print("Scaler saved.")
