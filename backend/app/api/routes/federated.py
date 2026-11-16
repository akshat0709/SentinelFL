import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import security_scheme, get_current_user

from app.schemas.federated import (
    FLStatusResponse,
    FLTrainRequest,
    FLJobStatusResponse,
    FLRunDetailResponse,
)
from app.services.fl_service import fl_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/federated", tags=["Federated Learning"])


@router.get("/status", response_model=FLStatusResponse)
async def get_federated_status():
    """
    Returns the real-time operational status of the Federated Learning engine,
    including available data partitioning scenarios and client counts.
    """
    return fl_service.get_status()


@router.get("/runs", response_model=List[Dict[str, Any]])
async def list_completed_fl_runs():
    """
    Returns the registry of historical Federated Learning runs, communication round counts,
    and convergence metrics from disk.
    """
    return fl_service.list_completed_runs()


@router.get("/runs/{run_id}", response_model=FLRunDetailResponse)
async def get_fl_run_details(run_id: str):
    """
    Returns communication round convergence history, participating edge client evaluations,
    and hyperparameters for the specified run.
    """
    try:
        details = fl_service.get_run_details(run_id)
        return details
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as exc:
        logger.error(f"Error fetching run details for {run_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/train", response_model=FLJobStatusResponse)
async def start_federated_training(
    req: FLTrainRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
):
    """
    Launches a genuine Federated Learning (FedAvg) training job in the background.
    With ``dataset_id`` the selected uploaded dataset is cleaned, split into a held-out test set
    and client partitions, and trained on; otherwise the built-in CIC-IoT-2023 benchmark cache
    and its fixed split files are used.
    """
    source = None
    if req.dataset_id:
        user = await get_current_user(credentials)
        from app.api.routes.models import load_dataset_bytes_for_user
        raw, name = load_dataset_bytes_for_user(req.dataset_id, user)
        try:
            source = fl_service.prepare_dataset_source(raw, name, req)
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    try:
        status_res = fl_service.start_training_job(req, source=source)
        return status_res
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(re))
    except (ValueError, FileNotFoundError) as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.error(f"Error initiating FL training: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/jobs/{job_id}", response_model=FLJobStatusResponse)
async def get_fl_job_status(job_id: str):
    """
    Retrieves progress telemetry, round history, and current status of an active or recent FL training job.
    """
    try:
        return fl_service.get_job_status(job_id)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
