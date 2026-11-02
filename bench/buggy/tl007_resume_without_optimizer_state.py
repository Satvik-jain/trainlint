# expect: TL007
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X, y = torch.randn(512, 20), torch.randint(0, 2, (512,))
train_loader = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=64)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

parser = argparse.ArgumentParser()
parser.add_argument("--resume")
args = parser.parse_args()

start_epoch = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

for epoch in range(start_epoch, 10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "last.pt")
