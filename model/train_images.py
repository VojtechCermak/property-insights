import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# Configuration
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Arguments and constants
epochs = 500
lr = 0.01
batch_size = 256
cut_quantile = 0.8

# Load metadata
meta = pd.read_csv('../data/processed/metadata.csv')
price_cut = meta['price'].quantile(cut_quantile)
meta = meta[meta['price'] < price_cut]

# Objective setup
# Always use lognorm_price_psm
y_raw = np.log(meta['price_per_sqm'].values)
mean, std = y_raw.mean(), y_raw.std()
y = (y_raw - mean) / std
def inv_transform(y, X):
    return np.exp(y * std + mean) * X['usableArea'].values

X = meta.copy()
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Load image features
with open('/home/vojta/projects/real-broker/data/processed/image_features.pkl', 'rb') as f:
    img_feats = pickle.load(f)

df_feat = pd.DataFrame(img_feats['features'])
df_feat = df_feat.groupby(img_feats['property_ids']).mean()

X_img_trainval = df_feat.loc[X_trainval['id']].values
X_img_test = df_feat.loc[X_test['id']].values

# Dataset class
class PropertyDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

trainval_dataset = PropertyDataset(X_img_trainval, y_trainval)
test_dataset = PropertyDataset(X_img_test, y_test)
trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)

# Model
def get_model(input_dim, dropout=0.5):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.Sigmoid(),
        nn.Dropout(dropout),
        nn.BatchNorm1d(512),
        nn.Linear(512, 1)
    )

def train_model(X_train, y_train, X_val=None, y_val=None, epochs=500, batch_size=256, lr=0.01, input_dim=None):
    model = get_model(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    loss_fn = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.0001)
    train_dataset = PropertyDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if X_val is not None and y_val is not None:
        val_dataset = PropertyDataset(X_val, y_val)
    best_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if X_val is not None and y_val is not None:
                val_preds = model(val_dataset.X)
                val_loss = loss_fn(val_preds, val_dataset.y).item()
            else:
                val_preds = model(train_dataset.X)
                val_loss = loss_fn(val_preds, train_dataset.y).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
        if (epoch+1) % 10 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")
        scheduler.step()
    model.load_state_dict(best_state)
    model.eval()
    return model

def eval_model(model, X, y, inv_transform, X_raw):
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy().flatten()
    preds_inv = inv_transform(preds, X_raw)
    y_inv = inv_transform(y, X_raw)
    metrics = {
        'RMSE': float(mean_squared_error(y_inv, preds_inv, squared=False)),
        'MAPE': float(mean_absolute_percentage_error(y_inv, preds_inv)),
        'MAE': float(mean_absolute_error(y_inv, preds_inv)),
        'Median_AE': float(np.median(np.abs(preds_inv - y_inv)))
    }
    return preds, preds_inv, y_inv, metrics

cv = KFold(n_splits=4, shuffle=True, random_state=SEED)

# Prepare arrays for OOF predictions
oof_preds = np.zeros_like(y_trainval)
metrics_folds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_trainval)):
    print(f"Fold {fold+1}/4")
    X_tr, X_val = X_img_trainval[train_idx], X_img_trainval[val_idx]
    y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]
    model = train_model(X_tr, y_tr, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=lr, input_dim=X_img_trainval.shape[1])
    # OOF predictions
    preds_val, preds_val_inv, y_val_inv, metrics = eval_model(model, X_val, y_val, inv_transform, X_trainval.iloc[val_idx])
    oof_preds[val_idx] = preds_val
    metrics_folds.append(metrics)
    print(f"Fold {fold+1} metrics:", metrics)

# Inverse transform
oof_preds_inv = inv_transform(oof_preds, X_trainval)
y_trainval_inv = inv_transform(y_trainval, X_trainval)
y_test_inv = inv_transform(y_test, X_test)

# Save results for stacking
os.makedirs('artifacts/image_dense', exist_ok=True)
pd.DataFrame({
    'id': X_trainval['id'].values,
    'y_true': y_trainval_inv,
    'y_pred': oof_preds_inv
    }).to_csv('artifacts/image_dense/predictions_oof.csv', index=False)

with open('artifacts/image_dense/metrics.json', 'w') as f:
    json.dump({'folds': metrics_folds}, f, indent=2)

# Retrain on all trainval for final test predictions
print("Retraining on all trainval data for final test predictions...")
model = train_model(X_img_trainval, y_trainval, epochs=epochs, batch_size=batch_size, lr=lr, input_dim=X_img_trainval.shape[1])
preds_test_full, preds_test_full_inv, _, metrics_full = eval_model(model, X_img_test, y_test, inv_transform, X_test)
print("Full trainval model test metrics:", metrics_full)
pd.DataFrame({
    'id': X_test['id'].values,
    'y_true': y_test_inv,
    'y_pred': preds_test_full_inv
    }).to_csv('artifacts/image_dense/predictions_test_full.csv', index=False)
with open('artifacts/image_dense/metrics_full.json', 'w') as f:
    json.dump(metrics_full, f, indent=2)
