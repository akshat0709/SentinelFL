"""Training loops: centralized, federated (FedAvg) and local-only — all on the SAME
model, initial weights, scaler, local optimiser (fl_core.train_local) and checkpoint
selection rule (best validation macro-F1), so comparisons are fair.

FedAvg here is the plain PyTorch equivalent of the Flower strategy used in FLipynb
(sample-weighted mean of full state_dicts, all clients every round); the Colab run
verified the two agree to 1.19e-7. Seeds per round/client follow FLipynb cell 7:
``seed + 1000 * round + client``.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .metrics import confusion, metrics_from_cm
from .model import FLMLP, payload_bytes, predict_logits, train_local, weighted_average


@dataclass
class ClientData:
    X_train: np.ndarray   # already scaled, float32
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    @property
    def n_train(self) -> int:
        return int(len(self.y_train))


def predict(model, X) -> np.ndarray:
    return predict_logits(model, X).argmax(1).numpy()


def evaluate_cm(model, X, y) -> np.ndarray:
    return confusion(y, predict(model, X))


def _clone_state(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _log(log_fn, msg):
    if log_fn:
        log_fn(msg)


# ------------------------------------------------------------------------- FedAvg
def run_fedavg(clients: List[ClientData], init_state: Dict, rounds: int = 20, local_epochs: int = 1,
               batch_size: int = 512, lr: float = 1e-3, seed: int = 42,
               log_fn: Optional[Callable[[str], None]] = None) -> Dict:
    model = FLMLP()
    model.load_state_dict(init_state)
    per_transfer = payload_bytes(model)
    K = len(clients)
    weights_n = [c.n_train for c in clients]
    history, best = [], {"macro_f1": -1.0, "round": 0, "state": _clone_state(model)}
    bytes_total = 0
    t_start = time.time()
    for r in range(1, rounds + 1):
        t0 = time.time()
        global_state = _clone_state(model)
        local_states, losses, client_secs = [], [], []
        for k, c in enumerate(clients):
            local = FLMLP()
            local.load_state_dict(global_state)
            tc = time.time()
            m = train_local(local, c.X_train, c.y_train, epochs=local_epochs, batch_size=batch_size,
                            lr=lr, seed=seed + 1000 * r + k)
            client_secs.append(time.time() - tc)
            local_states.append(_clone_state(local))
            losses.append(m["train_loss"])
        t_agg = time.time()
        model.load_state_dict(weighted_average(local_states, weights_n))
        agg_secs = time.time() - t_agg
        bytes_round = 2 * K * per_transfer          # global down + local up, every client
        bytes_total += bytes_round
        cms = [evaluate_cm(model, c.X_val, c.y_val) for c in clients]
        val = metrics_from_cm(np.sum(cms, axis=0))
        per_client_f1 = [metrics_from_cm(cm)["macro_f1"] for cm in cms]
        rec = {
            "round": r,
            "train_loss": float(np.average(losses, weights=weights_n)),
            "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "val_min_client_macro_f1": float(min(per_client_f1)),
            "bytes_round": int(bytes_round), "bytes_cumulative": int(bytes_total),
            "round_secs": time.time() - t0, "max_client_secs": float(max(client_secs)),
            "aggregation_secs": agg_secs,
        }
        history.append(rec)
        if val["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val["macro_f1"], "round": r, "state": _clone_state(model)}
        _log(log_fn, f"  round {r:2d}/{rounds} val_acc={val['accuracy']:.4f} "
                     f"val_macroF1={val['macro_f1']:.4f} ({rec['round_secs']:.1f}s)")
    return {"history": history, "best_state": best["state"], "best_round": best["round"],
            "best_val_macro_f1": best["macro_f1"], "final_state": _clone_state(model),
            "payload_bytes_per_transfer": per_transfer, "bytes_total": bytes_total,
            "wall_secs": time.time() - t_start}


# --------------------------------------------------------------------- centralized
def run_centralized(X_train, y_train, X_val, y_val, init_state: Dict, epochs: int = 20,
                    batch_size: int = 512, lr: float = 1e-3, seed: int = 42,
                    log_fn: Optional[Callable[[str], None]] = None) -> Dict:
    """Pooled training with the identical training function (Adam re-created each pass,
    exactly like FLipynb cell 8). One epoch = one pass = one FL round at E=1."""
    model = FLMLP()
    model.load_state_dict(init_state)
    history, best = [], {"macro_f1": -1.0, "epoch": 0, "state": _clone_state(model)}
    t_start = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        m = train_local(model, X_train, y_train, epochs=1, batch_size=batch_size, lr=lr, seed=seed + 1000 * ep)
        val = metrics_from_cm(evaluate_cm(model, X_val, y_val))
        history.append({"epoch": ep, "train_loss": m["train_loss"], "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "epoch_secs": time.time() - t0})
        if val["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val["macro_f1"], "epoch": ep, "state": _clone_state(model)}
        _log(log_fn, f"  epoch {ep:2d}/{epochs} val_acc={val['accuracy']:.4f} val_macroF1={val['macro_f1']:.4f}")
    return {"history": history, "best_state": best["state"], "best_epoch": best["epoch"],
            "best_val_macro_f1": best["macro_f1"], "final_state": _clone_state(model),
            "bytes_total": None, "wall_secs": time.time() - t_start}


# ---------------------------------------------------------------------- local-only
def run_local_only(clients: List[ClientData], init_state: Dict, epochs: int = 20, batch_size: int = 512,
                   lr: float = 1e-3, seed: int = 42, log_fn=None) -> List[Dict]:
    """Each client trains alone on its own data (no federation); best on its own val."""
    out = []
    for k, c in enumerate(clients):
        _log(log_fn, f" local-only client {k}")
        res = run_centralized(c.X_train, c.y_train, c.X_val, c.y_val, init_state, epochs=epochs,
                              batch_size=batch_size, lr=lr, seed=seed + 100 * k, log_fn=None)
        res["client"] = k
        out.append(res)
    return out


def bytes_raw_data(n_rows: int, n_features: int = 39, bytes_per_value: int = 4) -> int:
    """Bytes needed to ship raw feature rows to a central server (for the comms comparison)."""
    return int(n_rows) * n_features * bytes_per_value
