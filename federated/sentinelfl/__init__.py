"""sentinelfl — shared, reusable pipeline for SentinelFL.

One dataset adapter + one preprocessing contract used by EVERY consumer:
centralized training, federated (FedAvg) training, the FastAPI backend, and Colab.

    raw CSV --adapter.align()--> canonical 39-feature frame + canonical labels
            --clean_for_training() / impute_for_inference()-->
            --scaler.transform()--> FLMLP --> 7-class prediction

Import note: this package lives in ``federated/`` next to ``fl_core.py``. Put the
``federated/`` directory on ``sys.path`` (the backend and the scripts do this) and use
``import sentinelfl``.
"""
from .contract import FEATURES, CLASSES, CLASS_TO_ID, LABEL_ALIASES, normalize_label
from .adapters import (
    AlignedData,
    DatasetCompatibilityError,
    CICIoT2023Adapter,
    get_adapter,
    clean_for_training,
    impute_for_inference,
)
from .scalers import Log1pStandardScaler, local_stats, aggregate_stats
from .bundle import ModelBundle

__all__ = [
    "FEATURES", "CLASSES", "CLASS_TO_ID", "LABEL_ALIASES", "normalize_label",
    "AlignedData", "DatasetCompatibilityError", "CICIoT2023Adapter", "get_adapter",
    "clean_for_training", "impute_for_inference",
    "Log1pStandardScaler", "local_stats", "aggregate_stats",
    "ModelBundle",
]
__version__ = "1.0.0"
