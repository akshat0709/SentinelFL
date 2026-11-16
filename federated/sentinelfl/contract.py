"""Canonical data contract: the 39 input features and 7 classes.

The single source of truth stays in ``fl_core.py`` (it defines the model the existing
checkpoints were trained with). This module re-exports it and adds label aliases so
raw CIC IoT 2023 label names map onto the 7 canonical classes.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

_FED_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FED_DIR not in sys.path:
    sys.path.append(_FED_DIR)

import fl_core  # noqa: E402  (federated/fl_core.py)

FEATURES: List[str] = list(fl_core.FEATURES)
CLASSES: List[str] = list(fl_core.CLASSES)
CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

assert len(FEATURES) == 39 and len(set(FEATURES)) == 39, "feature contract must be 39 unique names"
assert len(CLASSES) == 7, "class contract must have 7 classes"

# Lower-cased raw label -> canonical class. Includes the canonical names themselves,
# the raw CIC IoT 2023 names, and a few common spellings seen in exports.
LABEL_ALIASES: Dict[str, str] = {c.lower(): c for c in CLASSES}
LABEL_ALIASES.update({
    "benigntraffic": "Benign",
    "normal": "Benign",
    "0": "Benign",
    "ddos-ack_fragmentation": "DDoS-ACK",
    "ddos-ack": "DDoS-ACK",
    "ddos-syn_flood": "DDoS-SYN",
    "ddos-syn": "DDoS-SYN",
    "ddos-tcp_flood": "DDoS-TCP",
    "ddos-tcp": "DDoS-TCP",
    "mirai-greeth_flood": "Mirai-greeth_flood",
    "mirai-greip_flood": "Mirai-greip_flood",
    "mirai-udpplain": "Mirai-udpplain",
})

LABEL_COLUMN_CANDIDATES = ["Label", "label", "attack_cat", "attack_type", "class", "target", "category"]


def normalize_label(raw) -> Optional[str]:
    """Map a raw label to a canonical class name, or None if it is not one of the 7 classes."""
    if raw is None:
        return None
    try:
        import math
        if isinstance(raw, float) and math.isnan(raw):
            return None
    except Exception:  # pragma: no cover
        pass
    return LABEL_ALIASES.get(str(raw).strip().lower())
