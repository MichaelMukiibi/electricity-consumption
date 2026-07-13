import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Dataset Construction Abstraction
class ElectricityDataset(Dataset):
    """Prepares time-series window sequences and targets for PyTorch models."""
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# 2. Modular Reconstructed Model Architecture
class ElectricityForecaster(nn.Module):
    """Flexible recurrent network mapping historical windows to future consumption."""
    def __init__(self, cell_type, input_size, hidden_size, output_size):
        super().__init__()
        self.cell_type = cell_type
        
        if cell_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Select 'RNN', 'LSTM', or 'GRU'")
            
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 3. Data Pipeline Processing
def load_and_preprocess_data(seq_length=24):
    """Downloads, chronologically sorts, normalizes, and slices the PJME dataset."""
    print("Loading data from remote source...")
    DATA_URL = "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv"
    df = pd.read_csv(DATA_URL, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()
    
    raw_values = df['PJME_MW'].values.astype(np.float32)
    min_val, max_val = raw_values.min(), raw_values.max()
    normalized_values = (raw_values - min_val) / (max_val - min_val)
    
    xs, ys = [], []
    for i in range(len(normalized_values) - seq_length):
        xs.append(normalized_values[i:(i + seq_length)])
        ys.append(normalized_values[i + seq_length])
        
    return np.array(xs), np.array(ys)

# 4. Core Training Pipeline Execution Loop
def run_training(model, dataloader, device, epochs=3, lr=0.001):
    """Executes backpropagation, tracking MSE loss across epochs."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n--- Training {model.cell_type} Architecture ---")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for seqs, targets in dataloader:
            seqs, targets = seqs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Mean Loss: {total_loss/len(dataloader):.5f}")

if __name__ == '__main__':
    # Parse CLI configurations
    parser = argparse.ArgumentParser(description="PyTorch Time-Series Forecasting")
    parser.add_argument('--cell', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'], help="Recurrent backbone engine")
    parser.add_argument('--epochs', type=int, default=3, help="Total training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="DataLoader batch footprint size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate factor")
    args = parser.parse_args()

    # Runtime Environment Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing workflow pipeline on device: {device}")

    # Process and build DataLoaders
    X, y = load_and_preprocess_data(seq_length=24)
    
    # Use the full dataset instead of small subsamples for structural execution
    dataset = ElectricityDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Instantiate custom parameters
    forecaster_model = ElectricityForecaster(
        cell_type=args.cell, 
        input_size=1, 
        hidden_size=32, 
        output_size=1
    )
    
    # Run pipeline loop
    run_training(
        model=forecaster_model, 
        dataloader=dataloader, 
        device=device, 
        epochs=args.epochs, 
        lr=args.lr
    )