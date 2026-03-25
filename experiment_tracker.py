import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import StandardScaler

# 1. Model Definition
class HousingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HousingModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

# 2. Data Loading & Preprocessing
df = pd.read_csv('data/housing.csv')
X = df[['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']].values
y = df['price_jod'].values.reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 3. Train/Test Split (80/20)
torch.manual_seed(42)
indices = torch.randperm(len(X_tensor))
split = int(0.8 * len(X_tensor))

X_train, X_test = X_tensor[indices[:split]], X_tensor[indices[split:]]
y_train, y_test = y_tensor[indices[:split]], y_tensor[indices[split:]]

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

# 5. Experiment Grid Design
lrs = [0.1, 0.01, 0.001]
hidden_sizes = [16, 32, 64, 128]
epochs_list = [50, 100, 200]
results = []

print(f"{'Rank':<5} | {'LR':<7} | {'Hidden':<6} | {'Epochs':<6} | {'MAE':<10} | {'R2':<7} | {'Time'}")
print("-" * 65)

# 6. Execution Loop
for lr, hidden, epochs in product(lrs, hidden_sizes, epochs_list):
    start_time = time.time()
    
    model = HousingModel(input_size=5, hidden_size=hidden)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_scaled)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_scaled)
        mae = torch.mean(torch.abs(preds - y_test)).item()
        
        y_test_mean = torch.mean(y_test)
        ss_res = torch.sum((y_test - preds) ** 2)
        ss_tot = torch.sum((y_test - y_test_mean) ** 2)
        r2 = (1 - (ss_res / ss_tot)).item()

    elapsed = time.time() - start_time
    results.append({
        "lr": lr, "hidden_size": hidden, "epochs": epochs,
        "mae": mae, "r2": r2, "time": round(elapsed, 4)
    })

# 7. Sort and Print Leaderboard
results.sort(key=lambda x: x['mae'])
for i, res in enumerate(results[:10]):
    print(f"{i+1:<5} | {res['lr']:<7} | {res['hidden_size']:<6} | {res['epochs']:<6} | {res['mae']:<10.2f} | {res['r2']:<7.4f} | {res['time']}s")

# 8. Log to JSON
with open('experiments.json', 'w') as f:
    json.dump(results, f, indent=4)

# 9. Summary Visualization
plt.figure(figsize=(10, 6))
for h in hidden_sizes:
    subset = [r for r in results if r['hidden_size'] == h]
    subset.sort(key=lambda x: x['lr'])
    plt.plot([str(r['lr']) for r in subset], [r['mae'] for r in subset], marker='o', label=f'Hidden: {h}')

plt.title('Hyperparameter Search: MAE vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Test MAE (JOD)')
plt.legend()
plt.grid(True)
plt.savefig('experiment_summary.png')
print("\n✅ Saved experiments.json and experiment_summary.png")