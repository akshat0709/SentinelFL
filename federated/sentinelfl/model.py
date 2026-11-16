"""Model + training primitives, re-exported from fl_core (single source of truth)."""
from __future__ import annotations

from .contract import fl_core

FLMLP = fl_core.FLMLP
train_local = fl_core.train_local
predict_logits = fl_core.predict_logits
set_seed = fl_core.set_seed
weighted_average = fl_core.weighted_average


def n_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def payload_bytes(model) -> int:
    """Bytes of one full model transfer (all parameters, as stored)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
