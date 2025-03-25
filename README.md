# Higgs Boson Classification Using ML

This project uses real and simulated particle physics data to train a machine learning model that distinguishes Higgs boson events from background. It was developed as part of an MIT course project.
This is a simplified version of the full code to demonstrate the ideas, and all the data files are either removed or reduced. If you with to have access to the full code, email me.

## Goals
- Apply physics-inspired feature engineering and event selection ("cuts")
- Train a neural network to classify Higgs vs background
- Compare results with standard cut-based methods
- Improve signal significance using ML-based selection

## Structure
higgs-boson-ml/

├── CERN-ORIGINAL/              # Raw datasets from CERN

├── PROCESSED-DATA/             # Transformed training-ready data (from transform_data_all.py)

├── outputs/                    # Final plots and data after cut application

├── neural_network/             # Neural network model training, testing, and cut application

│   └── model_training.py       # (example) training and evaluation logic

├── src/                        # Helper functions used across scripts

├── data_analysis.py            # Main script to generate final plots after NN cuts

├── transform_data_all.py       # Script to process raw CERN data to NN-compatible format

├── README.md

└── requirements.txt

## Key Libraries
- PyTorch, Pandas, NumPy, Matplotlib

## 📊 Results

The neural network was able to isolate Higgs-like events with improved signal significance compared to traditional cut-based methods. Below is an example of the classification output:

![image](https://github.com/user-attachments/assets/e04bc553-7a99-4f44-993a-7bacf380f35c)


## 🚀 How to Run

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/higgs-boson-ml.git
cd higgs-boson-ml
pip install -r requirements.txt
