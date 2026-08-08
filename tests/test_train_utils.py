"""Smoke tests for train_utils.py.

These don't need the (git-ignored) telemetry .pkl datasets, so they can
run in CI on every push/PR.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_utils import EarlyStopping, get_device, iterate_minibatches, report_best_scores, set_seed


def test_set_seed_is_reproducible():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_iterate_minibatches_covers_all_rows():
    x = torch.arange(10).reshape(10, 1)
    y = torch.arange(10)
    seen = set()
    for bx, by in iterate_minibatches(x, y, batch_size=3, shuffle=False):
        seen.update(bx.flatten().tolist())
    assert seen == set(range(10))


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=2)
    model = torch.nn.Linear(1, 1)
    assert es.step(1.0, model, epoch=1) is False
    assert es.step(1.0, model, epoch=2) is False
    assert es.step(1.0, model, epoch=3) is True


def test_report_best_scores_picks_min_val_loss():
    history = {"val_loss": [0.5, 0.2, 0.3], "val_mae": [0.4, 0.1, 0.2]}
    best_epoch, best_val_loss, best_secondary = report_best_scores(history, "mae")
    assert best_epoch == 2
    assert best_val_loss == 0.2
    assert best_secondary == 0.1