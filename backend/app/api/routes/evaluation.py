import io
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, Response
from fastapi.responses import PlainTextResponse
from app.core.security import get_current_user, AuthenticatedUser
from app.database.supabase import verify_dataset_access
from app.services.ml_service import ml_service
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["Evaluation & Benchmarks"])


CLASSES = [
    "Benign",
    "DDoS-ACK",
    "DDoS-SYN",
    "DDoS-TCP",
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
]


@router.get("/benchmarks", response_model=List[Dict[str, Any]])
async def get_benchmark_experiments():
    """
    Returns benchmark comparisons (centralized vs FedAvg vs local-only vs XGBoost, per data
    scenario) produced by ``federated/sentinelfl/experiments.py`` and written to
    ``<EXPERIMENT_RESULTS_DIR>/benchmarks.json``.

    Every number served here is computed from saved experiment runs (best-validation
    checkpoint, evaluated once on the held-out test split). If no results file exists, an
    empty list is returned — the frontend then shows an explicit empty state instead of
    placeholder figures.
    """
    path = settings.EXPERIMENT_RESULTS_DIR / "benchmarks.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.error(f"Could not read benchmark results at {path}: {exc}")
        raise HTTPException(status_code=500, detail="Benchmark results file is unreadable.")
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="Benchmark results file has an unexpected format.")
    return data


@router.get("/dataset/{dataset_id}")
async def get_dataset_evaluation(
    dataset_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Returns empirical evaluation metrics and 7x7 confusion matrix for a specific dataset
    evaluated by the model. Only computed when genuine ground-truth labels are present.
    """
    try:
        verify_dataset_access(user_id=current_user.id, dataset_id=dataset_id, token=current_user.token)
    except TypeError:
        verify_dataset_access(user_id=current_user.id, dataset_id=dataset_id)

    eval_data = ml_service.get_dataset_evaluation(dataset_id)
    if not eval_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No evaluation results recorded for dataset '{dataset_id}'. Run batch classification on this dataset in Flow Detection first.",
        )

    return eval_data


@router.post("/evaluate-csv")
async def evaluate_csv(
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Receives an uploaded labelled CSV dataset, extracts features, isolates ground-truth labels,
    executes real batch inference through PyTorch FLMLP (mlp_model.pth), computes empirical metrics
    (Accuracy, Macro/Weighted F1, FPR, FNR, 7x7 confusion matrix, classification report),
    persists the evaluation run, and returns genuine results.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV dataset (.csv)",
        )

    try:
        content = await file.read()
        eval_result = ml_service.evaluate_csv_bytes(
            csv_bytes=content,
            filename=file.filename,
            model_id=model_id,
        )
        return eval_result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Failed to evaluate dataset '{file.filename}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(exc)}",
        )


@router.get("/history", response_model=List[Dict[str, Any]])
async def list_evaluation_history(
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Returns list of all persisted model evaluation runs from disk.
    Allows researchers to review past evaluation experiments.
    """
    return ml_service.list_saved_evaluations()


@router.get("/run/{eval_id}")
async def get_evaluation_by_id(
    eval_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Retrieves a specific evaluation run by its ID.
    """
    data = ml_service.get_saved_evaluation(eval_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{eval_id}' not found.",
        )
    return data


@router.get("/report/{eval_id}")
async def export_evaluation_report(
    eval_id: str,
    format: str = Query("json", pattern="^(json|markdown|csv)$"),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Exports an evaluation run as JSON, Markdown, or CSV report.
    """
    data = ml_service.get_saved_evaluation(eval_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{eval_id}' not found.",
        )

    if format == "json":
        return Response(
            content=json.dumps(data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={eval_id}_report.json"},
        )
    elif format == "markdown":
        ds_name = data.get("dataset_name") or data.get("datasetName", "Dataset")
        acc = data.get("accuracy", 0) * 100
        mf1 = data.get("macro_f1", 0) * 100
        wf1 = data.get("weighted_f1", 0) * 100
        mprec = data.get("macro_precision", 0) * 100
        mrec = data.get("macro_recall", 0) * 100
        fpr = data.get("false_positive_rate", 0) * 100
        fnr = data.get("false_negative_rate", 0) * 100
        n_samples = data.get("total_samples") or data.get("evaluated_samples", 0)
        dt = data.get("evaluated_at") or data.get("evaluatedAt", "")

        md = f"""# SentinelFL Model Evaluation Report

- **Evaluation ID:** `{eval_id}`
- **Dataset:** {ds_name}
- **Timestamp:** {dt}
- **Model:** {data.get('model_version') or data.get('modelVersion', 'FLMLP (39 -> 128 -> 64 -> 32 -> 7)')}
- **Evaluated Samples:** {n_samples:,}

## Summary Metrics

| Metric | Value |
|---|---|
| Accuracy | {acc:.2f}% |
| Macro F1-Score | {mf1:.2f}% |
| Weighted F1-Score | {wf1:.2f}% |
| Macro Precision | {mprec:.2f}% |
| Macro Recall | {mrec:.2f}% |
| False Positive Rate (FPR) | {fpr:.2f}% |
| False Negative Rate (FNR) | {fnr:.2f}% |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | FPR | Support |
|---|---|---|---|---|---|
"""
        for row in data.get("per_class_performance") or data.get("perClassPerformance", []):
            c_name = row.get("className") or row.get("class_name")
            p = (row.get("precision", 0) * 100)
            r = (row.get("recall", 0) * 100)
            f1 = (row.get("f1Score") or row.get("f1_score", 0)) * 100
            fp = (row.get("falsePositiveRate") or row.get("fpr", 0)) * 100
            s = row.get("support", 0)
            md += f"| {c_name} | {p:.2f}% | {r:.2f}% | {f1:.2f}% | {fp:.2f}% | {s:,} |\n"

        cm = data.get("confusion_matrix") or data.get("confusionMatrix", {})
        classes = cm.get("classes", [])
        matrix = cm.get("matrix", [])
        if classes and matrix:
            md += "\n## Confusion Matrix (Actual \\ Predicted)\n\n"
            md += "| Actual \\ Pred | " + " | ".join(classes) + " |\n"
            md += "|---" * (len(classes) + 1) + "|\n"
            for idx, row in enumerate(matrix):
                actual_name = classes[idx] if idx < len(classes) else f"Class {idx}"
                md += f"| **{actual_name}** | " + " | ".join(str(val) for val in row) + " |\n"

        return PlainTextResponse(
            content=md,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={eval_id}_report.md"},
        )
    elif format == "csv":
        csv_lines = ["class,precision,recall,f1_score,false_positive_rate,support"]
        for row in data.get("per_class_performance") or data.get("perClassPerformance", []):
            c_name = row.get("className") or row.get("class_name")
            p = row.get("precision", 0)
            r = row.get("recall", 0)
            f1 = row.get("f1Score") or row.get("f1_score", 0)
            fp = row.get("falsePositiveRate") or row.get("fpr", 0)
            s = row.get("support", 0)
            csv_lines.append(f'"{c_name}",{p},{r},{f1},{fp},{s}')
        return Response(
            content="\n".join(csv_lines),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={eval_id}_report.csv"},
        )

