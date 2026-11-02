import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X, y = torch.randn(512, 20), torch.randint(0, 2, (512,))
train_loader = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=64)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(loader.dataset)


for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(epoch, evaluate(model, val_loader))
