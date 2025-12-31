import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                           precision_recall_curve, average_precision_score,
                           precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from torch_optimizer import Ranger
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Parameters
SEQ_LEN = 24  # 24 hours lookback
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.001
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading with caching
def load_data(cache=True):
    cache_path = 'data/gold_data.feather'
    
    if cache and os.path.exists(cache_path):
        print("Loading cached data...")
        return pd.read_feather(cache_path)
    
    print("Downloading data...")
    data = yf.download("GC=F", period="730d", interval="1h")
    
    # Calculate technical indicators
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    print("Calculating features...")
    features = pd.DataFrame(index=data.index)
    
    # Price features
    features['close'] = data['Close']
    features['open'] = data['Open']
    features['high'] = data['High']
    features['low'] = data['Low']
    features['volume'] = data['Volume']
    
    # Moving averages
    features['sma_5'] = features['close'].rolling(5).mean()
    features['sma_10'] = features['close'].rolling(10).mean()
    features['sma_20'] = features['close'].rolling(20).mean()
    
    # Momentum indicators
    features['rsi'] = calculate_rsi(features['close'])
    features['macd'] = features['close'].ewm(span=12, adjust=False).mean() - features['close'].ewm(span=26, adjust=False).mean()
    features['momentum'] = features['close'].pct_change(5)
    
    # Volatility
    features['volatility'] = features['close'].pct_change().rolling(24).std()
    features['atr'] = (features['high'] - features['low']).rolling(14).mean()
    
    # Volume features
    features['volume_ma'] = features['volume'].rolling(5).mean()
    features['volume_spike'] = (features['volume'] / features['volume_ma']) - 1
    features['obv'] = (np.sign(features['close'].diff()) * features['volume']).cumsum()
    
    # Price ratios
    features['close_ratio'] = features['close'] / features['sma_10']
    features['range'] = (features['high'] - features['low']) / features['close']
    
    # Time features
    hour = features.index.hour
    dayofweek = features.index.dayofweek
    features['hour_sin'] = np.sin(2 * np.pi * hour/24)
    features['hour_cos'] = np.cos(2 * np.pi * hour/24)
    features['day_sin'] = np.sin(2 * np.pi * dayofweek/7)
    features['day_cos'] = np.cos(2 * np.pi * dayofweek/7)
    
    # Target - simple directional
    features['target'] = np.where(features['close'].shift(-1) > features['close'], 1, 0)
    features.dropna(inplace=True)
    
    if cache:
        features.reset_index().to_feather(cache_path)
    
    return features

# Data preparation
def prepare_data(data):
    feature_cols = ['rsi', 'macd', 'sma_5', 'sma_10', 'sma_20', 'momentum',
                   'volatility', 'atr', 'volume_spike', 'obv', 'close_ratio',
                   'range', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_cols])
    joblib.dump(scaler, 'models/gold_scaler.pkl')
    
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_sequences(scaled_features, data['target'].values, SEQ_LEN)
    return X_seq, y_seq, data['close']

# Simplified LSTM Model
class GoldPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(GoldPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take last timestep

# Training function
def train_model(X, y, close_prices):
    tscv = TimeSeriesSplit(n_splits=3)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n=== Training Fold {fold + 1} ===")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"Class distribution - Train: {Counter(y[train_idx])}, Val: {Counter(y[val_idx])}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        # Handle class imbalance
        class_counts = torch.bincount(y_train_t)
        weights = 1. / class_counts.float()
        sample_weights = weights[y_train_t]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=BATCH_SIZE)
        
        # Initialize model
        model = GoldPricePredictor(input_dim=X_train.shape[2]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2%}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'models/gold_lst_fold{fold+1}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered")
                    break
            
            scheduler.step(avg_val_loss)
        
        # Evaluate best model
        model.load_state_dict(torch.load(f'models/gold_lst_fold{fold+1}.pth'))
        model.eval()
        
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        cr = classification_report(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        ap = average_precision_score(all_targets, all_probs)
        
        fold_results.append({
            'classification_report': cr,
            'confusion_matrix': cm,
            'avg_precision': ap,
            'history': history
        })
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Fold {fold+1} - Loss History')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Accuracy')
        plt.title(f'Fold {fold+1} - Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/training_fold{fold+1}.png')
        plt.close()
    
    return fold_results

# Main execution
if __name__ == "__main__":
    try:
        print("Loading and preparing data...")
        data = load_data()
        X, y, close_prices = prepare_data(data)
        
        print("\nStarting training...")
        results = train_model(X, y, close_prices)
        
        print("\n=== Final Results ===")
        for i, res in enumerate(results):
            print(f"\nFold {i+1} Results:")
            print(res['classification_report'])
            print("\nConfusion Matrix:")
            print(res['confusion_matrix'])
            print(f"\nAverage Precision: {res['avg_precision']:.2%}")
        
        print("\nTraining complete. Models and plots saved.")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()