import pytest

# TL003 -----------------------------------------------------------------------


def test_eval_in_epoch_loop_without_train(lint):
    findings = lint(
        """
        model.train()
        for epoch in range(10):
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            model.eval()
            validate(model)
        """,
        "TL003",
    )
    assert [f.line for f in findings] == [8]


def test_eval_with_train_at_top_of_epoch_is_clean(lint):
    assert not lint(
        """
        for epoch in range(10):
            model.train()
            for x, y in loader:
                loss.backward()
            model.eval()
        """,
        "TL003",
    )


def test_eval_in_loop_that_does_not_train_is_clean(lint):
    assert not lint(
        """
        for epoch in range(10):
            train_one_epoch(model)
            model.eval()
            scheduler.step()
        """,
        "TL003",
    )


# TL004 -----------------------------------------------------------------------


def test_no_grad_without_eval(lint):
    findings = lint(
        """
        loss.backward()
        with torch.no_grad():
            preds = model(x)
        """,
        "TL004",
    )
    assert [f.line for f in findings] == [3]


def test_no_grad_decorator_without_eval(lint):
    findings = lint(
        """
        @torch.no_grad()
        def evaluate(model, loader):
            return [model(x) for x, _ in loader]

        loss.backward()
        """,
        "TL004",
    )
    assert len(findings) == 1


@pytest.mark.parametrize("call", ["model.eval()", "model.train(False)"])
def test_no_grad_with_eval_is_clean(lint, call):
    assert not lint(
        f"""
        loss.backward()
        {call}
        with torch.no_grad():
            preds = model(x)
        """,
        "TL004",
    )


def test_no_grad_file_that_does_not_train_is_clean(lint):
    assert not lint(
        """
        with torch.no_grad():
            preds = model(x)
        """,
        "TL004",
    )


# TL005 -----------------------------------------------------------------------


def test_unseeded_script(lint):
    findings = lint(
        """
        import torch

        def main():
            loss.backward()

        if __name__ == "__main__":
            main()
        """,
        "TL005",
    )
    assert [f.line for f in findings] == [5]


def test_unseeded_notebook_style_top_level_loop(lint):
    assert lint(
        """
        import torch
        for x in loader:
            loss.backward()
        """,
        "TL005",
    )


@pytest.mark.parametrize(
    "seed_line",
    ["torch.manual_seed(0)", "set_seed(0)", "pl.seed_everything(1)", "args = Args(seed=3)"],
)
def test_seeded_script_is_clean(lint, seed_line):
    assert not lint(
        f"""
        import torch
        {seed_line}
        for x in loader:
            loss.backward()
        """,
        "TL005",
    )


def test_library_module_without_entry_point_is_clean(lint):
    assert not lint(
        """
        import torch

        def train_step(loss):
            loss.backward()
        """,
        "TL005",
    )


# TL008 -----------------------------------------------------------------------


def test_softmax_before_cross_entropy(lint):
    findings = lint(
        """
        class Net(nn.Module):
            def forward(self, x):
                x = self.fc(x)
                return F.softmax(x, dim=1)

        criterion = nn.CrossEntropyLoss()
        """,
        "TL008",
    )
    assert [f.line for f in findings] == [5]


def test_sequential_ending_in_sigmoid_with_bce_logits(lint):
    findings = lint(
        """
        net = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        loss_fn = nn.BCEWithLogitsLoss()
        """,
        "TL008",
    )
    assert [f.line for f in findings] == [2]


def test_softmax_with_nll_or_bce_is_clean(lint):
    assert not lint(
        """
        class Net(nn.Module):
            def forward(self, x):
                return torch.sigmoid(self.fc(x))

        loss_fn = nn.BCELoss()
        """,
        "TL008",
    )


def test_softmax_only_on_the_eval_branch_is_clean(lint):
    assert not lint(
        """
        class Net(nn.Module):
            def forward(self, x):
                logits = self.fc(x)
                if not self.training:
                    return logits.softmax(-1)
                return logits

        criterion = nn.CrossEntropyLoss()
        """,
        "TL008",
    )


# TL009 -----------------------------------------------------------------------


def test_zero_grad_between_backward_and_step(lint):
    findings = lint(
        """
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        """,
        "TL009",
    )
    assert [f.line for f in findings] == [3]


def test_gradients_never_cleared(lint):
    findings = lint(
        """
        import torch
        for x, y in loader:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        """,
        "TL009",
    )
    assert [f.line for f in findings] == [5]


@pytest.mark.parametrize(
    "order",
    [
        "optimizer.zero_grad()\nloss.backward()\noptimizer.step()",
        "loss.backward()\noptimizer.step()\noptimizer.zero_grad()",
        "loss.backward()\noptimizer.step()\nfor p in model.parameters():\n    p.grad = None",
        "loss_d.backward()\nopt_d.step()\nopt_g.zero_grad()\nloss_g.backward()\nopt_g.step()",
    ],
)
def test_correct_gradient_handling_is_clean(lint, order):
    assert not lint("import torch\n" + order, "TL009")


# TL010 -----------------------------------------------------------------------


def test_running_loss_keeps_graph(lint):
    findings = lint(
        """
        for x, y in loader:
            loss = criterion(model(x), y)
            loss.backward()
            running_loss += loss
        """,
        "TL010",
    )
    assert [f.line for f in findings] == [5]


def test_appending_loss_tensor(lint):
    findings = lint(
        """
        def fit():
            loss.backward()
            history.append(loss)
        """,
        "TL010",
    )
    assert len(findings) == 1


def test_scaled_loss_is_tracked(lint):
    assert lint(
        """
        scaler.scale(loss).backward()
        total += loss * batch_size
        """,
        "TL010",
    )


@pytest.mark.parametrize(
    "line",
    [
        "total += loss.item()",
        "total += loss.detach()",
        "total += float(loss)",
        "total += loss.data",
    ],
)
def test_detached_loss_is_clean(lint, line):
    assert not lint(f"loss.backward()\n{line}\n", "TL010")


def test_accumulating_a_non_loss_tensor_is_clean(lint):
    assert not lint("loss.backward()\ncount += batch\n", "TL010")
