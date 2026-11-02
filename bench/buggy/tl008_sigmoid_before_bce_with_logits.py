# expect: TL008
import torch
from torch import nn

torch.manual_seed(0)
X, y = torch.randn(512, 20), torch.randint(0, 2, (512,)).float()

model = nn.Sequential(nn.Linear(20, 1), nn.Sigmoid())
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    loss = criterion(model(X).squeeze(1), y)
    loss.backward()
    optimizer.step()
