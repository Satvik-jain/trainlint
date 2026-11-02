import pytest

# TL006 -----------------------------------------------------------------------


def test_fp16_autocast_without_scaler(lint):
    findings = lint(
        """
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        """,
        "TL006",
    )
    assert [f.line for f in findings] == [2]


def test_explicit_float16_autocast_without_scaler(lint):
    assert lint(
        """
        with torch.autocast("cuda", dtype=torch.float16):
            pass
        loss.backward()
        """,
        "TL006",
    )


@pytest.mark.parametrize(
    "autocast",
    [
        'torch.autocast("cuda", dtype=torch.bfloat16)',
        'torch.autocast("cpu")',
        "torch.autocast(device_type=device, dtype=amp_dtype)",
        'torch.autocast("cuda", enabled=False)',
    ],
)
def test_autocast_that_needs_no_scaler_is_clean(lint, autocast):
    assert not lint(f"with {autocast}:\n    pass\nloss.backward()\n", "TL006")


def test_inference_only_autocast_is_clean(lint):
    assert not lint('with torch.autocast("cuda"):\n    preds = model(x)\n', "TL006")


def test_scaler_created_but_never_used(lint):
    findings = lint(
        """
        scaler = torch.amp.GradScaler()
        with torch.autocast("cuda"):
            loss = model(x)
        loss.backward()
        optimizer.step()
        """,
        "TL006",
    )
    assert [f.line for f in findings] == [2]


def test_scaled_loss_stepped_without_scaler(lint):
    findings = lint(
        """
        scaler = GradScaler()
        scaler.scale(loss).backward()
        optimizer.step()
        """,
        "TL006",
    )
    assert [f.line for f in findings] == [3]


def test_complete_scaler_loop_is_clean(lint):
    assert not lint(
        """
        scaler = GradScaler()
        with torch.autocast("cuda"):
            loss = model(x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        """,
        "TL006",
    )


# TL007 -----------------------------------------------------------------------

RESUME_SCRIPT = """
import torch
optimizer = torch.optim.Adam(model.parameters())
if args.resume:
    model.load_state_dict(torch.load(args.resume))
for epoch in range(n):
    {save}
"""


def test_resumable_script_saving_only_model_weights(lint):
    findings = lint(RESUME_SCRIPT.format(save="torch.save(model.state_dict(), 'ckpt.pt')"), "TL007")
    assert [f.line for f in findings] == [7]


def test_checkpoint_with_optimizer_state_is_clean(lint):
    save = "torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()}, 'c.pt')"
    assert not lint(RESUME_SCRIPT.format(save=save), "TL007")


def test_script_that_never_resumes_is_clean(lint):
    assert not lint(
        """
        optimizer = torch.optim.Adam(model.parameters())
        torch.save(model.state_dict(), "final.pt")
        """,
        "TL007",
    )


# TL011 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    ["f1_score(y_pred, y_test)", "recall_score(preds, labels)", "r2_score(pred, y)"],
)
def test_swapped_metric_arguments(lint, call):
    assert len(lint(call + "\n", "TL011")) == 1


def test_correct_metric_order_is_clean(lint):
    assert not lint("f1_score(y_test, y_pred)\naccuracy_score(y_pred, y_test)\n", "TL011")


def test_auc_from_predict_call(lint):
    findings = lint("auc = roc_auc_score(y_test, model.predict(X_test))\n", "TL011")
    assert [f.line for f in findings] == [1]


def test_auc_from_predict_variable(lint):
    findings = lint(
        """
        y_hat = clf.predict(X_test)
        auc = roc_auc_score(y_test, y_hat)
        """,
        "TL011",
    )
    assert [f.line for f in findings] == [3]


@pytest.mark.parametrize(
    "scores",
    ["clf.predict_proba(X)[:, 1]", "clf.decision_function(X)", "booster.predict(X)"],
)
def test_auc_from_scores_is_clean(lint, scores):
    assert not lint(f"roc_auc_score(y_test, {scores})\n", "TL011")


def test_auc_variable_reassigned_to_probabilities_is_clean(lint):
    assert not lint(
        """
        scores = clf.predict(X)
        scores = clf.predict_proba(X)[:, 1]
        roc_auc_score(y, scores)
        """,
        "TL011",
    )


# TL012 -----------------------------------------------------------------------


def test_validation_loader_drops_last(lint):
    findings = lint("val_loader = DataLoader(val_ds, batch_size=64, drop_last=True)\n", "TL012")
    assert [f.line for f in findings] == [1]


def test_loader_named_by_dataset_drops_last(lint):
    assert lint("loader = DataLoader(test_dataset, drop_last=True)\n", "TL012")


def test_training_loader_may_drop_last(lint):
    assert not lint(
        """
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, drop_last=False)
        """,
        "TL012",
    )
