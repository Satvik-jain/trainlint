# Rules

Each rule has one example that is flagged and one that is not. The test suite runs both, so the
examples here are what the checker does.

## TL001: preprocessing fitted on all rows before the split

A scaler, imputer, encoder, feature selector or resampler learns from the rows it is fitted on.
Fitting it on everything and splitting afterwards lets validation and test rows influence the
transform. Feature selection is the worst case: on random data it turns a 50% score into 85% or
more.

```python
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
```

For cross-validation, put the step in a `Pipeline` and cross-validate the pipeline. Encoding the
label column on the full dataset is not flagged.

## TL002: transform or model fitted on a validation or test split

Fitting on held-out data means the "held-out" rows were used for training. The rule goes by
variable names such as `X_test`, `val_df` and `y_valid`.

```python
X_test = scaler.fit_transform(X_test)
```

```python
X_test = scaler.transform(X_test)
```

## TL003: eval() inside the training loop with no train() to switch back

`model.eval()` turns dropout off and makes batch norm use running statistics. If `model.train()`
is called once before the epoch loop, every epoch after the first validation trains in eval mode.

```python
model.train()
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    model.eval()
    validate(model, val_loader)
```

```python
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    model.eval()
    validate(model, val_loader)
```

## TL004: evaluation under no_grad but the model is never put in eval mode

`torch.no_grad()` stops gradient tracking and nothing else. Dropout stays active and batch norm
keeps updating its running statistics while you evaluate.

```python
for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = model(x_val)
```

```python
for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(x_val)
```

## TL005: training script never sets a random seed

Without a seed, two runs differ and a gain of half a point may be noise. The rule applies to
files that import `torch`, call `backward()`, and either have a `__main__` guard or run the loop
at the top level. Any call or keyword argument with "seed" in its name counts as seeding.

```python
import torch

for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward()
```

```python
import torch

torch.manual_seed(0)

for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward()
```

## TL006: mixed precision without a working GradScaler

float16 gradients underflow to zero without loss scaling. The rule flags float16 autocast in a
training file with no `GradScaler`, a `GradScaler` that is never used to scale, and a scaled loss
that is stepped with `optimizer.step()` instead of `scaler.step(optimizer)`. bfloat16 and CPU
autocast need no scaler and are not flagged.

```python
with torch.autocast("cuda", dtype=torch.float16):
    loss = criterion(model(x), y)
loss.backward()
optimizer.step()
```

```python
scaler = torch.amp.GradScaler("cuda")
with torch.autocast("cuda", dtype=torch.float16):
    loss = criterion(model(x), y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## TL007: resumable script saves checkpoints without optimizer state

If a script has a resume option but the checkpoint holds only model weights, momentum and
adaptive learning rates restart from zero. Scripts that only save final weights for inference
are not flagged.

```python
optimizer = torch.optim.Adam(model.parameters())
if args.resume:
    model.load_state_dict(torch.load(args.resume))
torch.save(model.state_dict(), "last.pt")
```

```python
optimizer = torch.optim.Adam(model.parameters())
if args.resume:
    state = torch.load(args.resume)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
torch.save(state, "last.pt")
```

## TL008: softmax or sigmoid applied before a loss that expects logits

`CrossEntropyLoss` applies log-softmax itself and `BCEWithLogitsLoss` applies the sigmoid itself.
Feeding them probabilities squashes the gradients and training stalls near the start. The rule
looks at the last statement of `forward()` and the last layer of an `nn.Sequential`.

```python
class Net(nn.Module):
    def forward(self, x):
        return torch.softmax(self.body(x), dim=1)


criterion = nn.CrossEntropyLoss()
```

```python
class Net(nn.Module):
    def forward(self, x):
        return self.body(x)


criterion = nn.CrossEntropyLoss()
```

## TL009: gradients never cleared, or cleared between backward and step

Gradients accumulate until they are cleared. Clearing them after `backward()` and before
`step()` discards the gradients you just computed. Distributed autograd keeps gradients per
context and is not flagged.

```python
loss.backward()
optimizer.zero_grad()
optimizer.step()
```

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## TL010: loss stored as a tensor, keeping the autograd graph alive

Adding the loss tensor to a running total, or appending it to a list, keeps the graph of every
iteration in memory. Memory grows until the process runs out.

```python
loss.backward()
running_loss += loss
```

```python
loss.backward()
running_loss += loss.item()
```

## TL011: metric called with swapped arguments or hard labels

scikit-learn metrics take `(y_true, y_pred)`. Precision, recall, F1 and R² give a different number
when the two are swapped, and nothing complains. `roc_auc_score` and `average_precision_score`
need scores; `predict()` output collapses the curve to a single point.

```python
f1 = f1_score(y_pred, y_test)
auc = roc_auc_score(y_test, model.predict(X_test))
```

```python
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

Calls on objects named like gradient boosting boosters (`booster`, `bst`, `xgb`, `lgb`) are not
flagged, because their `predict()` returns probabilities.

## TL012: evaluation DataLoader drops the last batch

`drop_last=True` discards the final partial batch, so a few validation or test samples are
never scored. It is fine for training loaders.

```python
val_loader = DataLoader(val_ds, batch_size=64, drop_last=True)
```

```python
val_loader = DataLoader(val_ds, batch_size=64, drop_last=False)
```

## TL013: data rescaled with its own statistics before the split

Scaling a dataset by its own mean, std, min or max before splitting it uses every row to pick the
scale. The validation and test rows then shape the training input, as with a scaler fitted on all
rows. The rule needs the rescaled name to be the one whose statistics it uses, and a split call
after it on the result.

```python
df = (df - df.mean()) / df.std()
X_train, X_test = train_test_split(df)
```

```python
X_train, X_test = train_test_split(df)
mean, std = X_train.mean(), X_train.std()
X_train, X_test = (X_train - mean) / std, (X_test - mean) / std
```
