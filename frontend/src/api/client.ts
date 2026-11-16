import { localSession } from '@/lib/localSession';
import { storage } from '@/lib/storage';
import { supabase } from '@/lib/supabase';
import { BackendStatus } from '@/types/settings';
import { Dataset } from '@/types/dataset';
import { ModelItem } from '@/types/model';
import { FLExperiment, FLClient, FLRoundSummary } from '@/types/federated';
import { NetworkFlowRecord, BatchClassificationResult } from '@/types/detection';
import { EvaluationExperiment } from '@/types/evaluation';
import { LogEntry } from '@/types/logs';

async function getAuthHeaders(): Promise<HeadersInit> {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    const headers: Record<string, string> = {
      'Accept': 'application/json',
    };
    const token = session?.access_token || localSession.get();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  } catch {
    return { 'Accept': 'application/json' };
  }
}


export interface ModelHealth {
  status: string;
  model_name: string;
  model_version: string;
  architecture: string;
  device: string;
  input_features_count: number;
  num_classes: number;
  classes: string[];
  feature_columns: string[];
  scaler_type: string;
  checkpoint_loaded: boolean;
  error?: string | null;
}

export interface SingleFlowPredictPayload {
  flow_id?: string;
  src_ip?: string;
  dst_ip?: string;
  src_port?: number;
  dst_port?: number;
  protocol?: string;
  flow_duration_ms?: number;
  tot_fwd_pkts?: number;
  flags?: string;
  features?: Record<string, number>;
}

export interface FLTrainingConfig {
  scenario?: string;
  rounds?: number;
  local_epochs?: number;
  batch_size?: number;
  lr?: number;
  seed?: number;
  num_clients?: number;
  max_samples_per_client?: number;
  dataset_id?: string;
}

export class ApiClient {
  public get baseUrl(): string {
    const configured = storage.getSettings().backendApiUrl;
    if (configured && configured !== 'http://localhost:8000') {
      return configured.replace(/\/+$/, '');
    }
    const envUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
    if (envUrl) {
      return envUrl.replace(/\/+$/, '');
    }
    // If deployed on remote host or unified container, default to current origin
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      return window.location.origin;
    }
    return configured || 'http://localhost:8000';
  }

  /**
   * Health check to detect if local FastAPI server is online.
   */
  async checkBackendHealth(): Promise<BackendStatus> {
    const start = performance.now();
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const res = await fetch(`${this.baseUrl}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const latency = Math.round(performance.now() - start);

      if (res.ok) {
        const data = await res.json().catch(() => ({}));
        const isOnline = data.status === 'healthy' || data.status === 'degraded' || data.readiness?.backend === true;
        return {
          connected: isOnline,
          service: data.service || 'FastAPI Backend',
          version: data.version || '1.0.0',
          latencyMs: latency,
          lastChecked: new Date().toISOString(),
          error: !isOnline ? 'Inference engine not ready' : undefined,
        };
      } else {
        return {
          connected: false,
          service: 'FastAPI Backend',
          latencyMs: latency,
          lastChecked: new Date().toISOString(),
          error: `HTTP ${res.status}: ${res.statusText}`,
        };
      }
    } catch {
      return {
        connected: false,
        service: 'FastAPI Backend',
        lastChecked: new Date().toISOString(),
        error: 'Connection refused (server offline or unreachable)',
      };
    }
  }

  /**
   * Returns operational state of the centralized MLP model.
   */
  async getModelHealth(): Promise<ModelHealth | null> {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/models/health`);
      if (res.ok) {
        return await res.json();
      }
      return null;
    } catch (e) {
      console.warn('Failed to query /api/v1/models/health:', e);
      return null;
    }
  }

  /**
   * Retrieves registered models from FastAPI backend with fallback to storage.
   */
  async getModels(): Promise<{ data: ModelItem[]; isBackend: boolean }> {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/models/`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          return { data, isBackend: true };
        }
      }
    } catch (e) {
      console.warn('Backend /api/v1/models/ unreachable, falling back to local storage:', e);
    }

    return {
      data: storage.getModels(false),
      isBackend: false,
    };
  }

  /**
   * Generates genuine inference prediction for a single flow record.
   */
  async predictSingleFlow(payload: SingleFlowPredictPayload): Promise<NetworkFlowRecord> {
    const res = await fetch(`${this.baseUrl}/api/v1/models/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Inference failed with status ${res.status}`);
    }

    const data = await res.json();
    return {
      id: data.id,
      timestamp: data.timestamp,
      srcIp: data.src_ip ?? payload.src_ip ?? null,
      srcPort: data.src_port ?? payload.src_port ?? null,
      dstIp: data.dst_ip ?? payload.dst_ip ?? null,
      dstPort: data.dst_port ?? payload.dst_port ?? null,
      protocol: (data.protocol ?? payload.protocol ?? null) as any,
      flowDurationMs: data.flow_duration_ms ?? payload.flow_duration_ms ?? null,
      totFwdPkts: data.tot_fwd_pkts ?? payload.tot_fwd_pkts ?? null,
      totBwdPkts: null,
      pktLenMean: null,
      flowBytsPerSec: null,
      flags: data.flags ?? payload.flags ?? null,
      keyFeatures: data.key_features ?? null,
      predictedClass: data.predicted_class,
      isMalicious: data.is_malicious,
      confidence: data.confidence,
      riskLevel: data.risk_level,
      modelVersion: data.model_version,
      isSyntheticSample: false,
    };
  }

  /**
   * Uploads a CSV for real-time batch classification by the MLP model.
   */
  async predictCsv(file: File, modelId?: string, forceIncompatible = false): Promise<BatchClassificationResult> {
    const headers = await getAuthHeaders();
    const formData = new FormData();
    formData.append('file', file);

    const url = new URL(`${this.baseUrl}/api/v1/models/predict-csv`);
    if (modelId) {
      url.searchParams.set('model_id', modelId);
    }
    if (forceIncompatible) {
      url.searchParams.set('force_incompatible', 'true');
    }

    const res = await fetch(url.toString(), {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `CSV classification failed with status ${res.status}`);
    }

    const data = await res.json();
    const flows: NetworkFlowRecord[] = (data.flows || []).map((f: any) => ({
      id: f.id,
      flowId: f.flow_id || f.id,
      timestamp: f.timestamp,
      srcIp: f.src_ip,
      srcPort: f.src_port,
      dstIp: f.dst_ip,
      dstPort: f.dst_port,
      protocol: f.protocol,
      flowDurationMs: f.flow_duration_ms,
      totFwdPkts: f.tot_fwd_pkts,
      totBwdPkts: f.tot_bwd_pkts ?? null,
      pktLenMean: f.pkt_len_mean ?? null,
      flowBytsPerSec: f.flow_byts_per_sec ?? null,
      keyFeatures: f.key_features ?? null,
      flags: f.flags,
      predictedClass: f.predicted_class,
      actualLabel: f.actual_label,
      status: f.status || 'classified',
      isMalicious: f.is_malicious,
      confidence: f.confidence,
      riskLevel: f.risk_level,
      modelVersion: f.model_version,
      isSyntheticSample: false,
    }));

    return {
      total_flows: data.total_flows,
      malicious_count: data.malicious_count,
      benign_count: data.benign_count,
      failed_count: data.failed_count || 0,
      flows,
      model_version: data.model_version,
      dataset_id: data.dataset_id,
      dataset_name: data.dataset_name,
      per_class_counts: data.per_class_counts || {},
      per_class_percentages: data.per_class_percentages || {},
      evaluation: data.evaluation,
      forced_compatibility: data.forced_compatibility || false,
      imputed_features: data.imputed_features || [],
      compatibility_warning: data.compatibility_warning || null,
    };
  }

  /**
   * Classifies all records in an uploaded workspace dataset directly on the backend.
   */
  async classifyDataset(datasetId: string, modelId?: string, forceIncompatible = false): Promise<BatchClassificationResult> {
    const headers = await getAuthHeaders();
    (headers as Record<string, string>)['Content-Type'] = 'application/json';

    const url = new URL(`${this.baseUrl}/api/v1/models/classify-dataset/${encodeURIComponent(datasetId)}`);
    if (modelId) {
      url.searchParams.set('model_id', modelId);
    }
    if (forceIncompatible) {
      url.searchParams.set('force_incompatible', 'true');
    }

    const res = await fetch(url.toString(), {
      method: 'POST',
      headers,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Dataset classification failed with status ${res.status}`);
    }

    const data = await res.json();
    const flows: NetworkFlowRecord[] = (data.flows || []).map((f: any) => ({
      id: f.id,
      flowId: f.flow_id || f.id,
      timestamp: f.timestamp,
      srcIp: f.src_ip,
      srcPort: f.src_port,
      dstIp: f.dst_ip,
      dstPort: f.dst_port,
      protocol: f.protocol,
      flowDurationMs: f.flow_duration_ms,
      totFwdPkts: f.tot_fwd_pkts,
      totBwdPkts: f.tot_bwd_pkts ?? null,
      pktLenMean: f.pkt_len_mean ?? null,
      flowBytsPerSec: f.flow_byts_per_sec ?? null,
      keyFeatures: f.key_features ?? null,
      flags: f.flags,
      predictedClass: f.predicted_class,
      actualLabel: f.actual_label,
      status: f.status || 'classified',
      isMalicious: f.is_malicious,
      confidence: f.confidence,
      riskLevel: f.risk_level,
      modelVersion: f.model_version,
      isSyntheticSample: false,
    }));

    return {
      total_flows: data.total_flows,
      malicious_count: data.malicious_count,
      benign_count: data.benign_count,
      failed_count: data.failed_count || 0,
      flows,
      model_version: data.model_version,
      dataset_id: data.dataset_id,
      dataset_name: data.dataset_name,
      per_class_counts: data.per_class_counts || {},
      per_class_percentages: data.per_class_percentages || {},
      evaluation: data.evaluation,
      forced_compatibility: data.forced_compatibility || false,
      imputed_features: data.imputed_features || [],
      compatibility_warning: data.compatibility_warning || null,
    };
  }

  /**
   * Retrieves previously cached batch predictions for an uploaded dataset.
   * Returns null if the dataset has not yet been classified.
   */
  async getDatasetPredictions(datasetId: string): Promise<BatchClassificationResult | null> {
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${this.baseUrl}/api/v1/models/predictions/${encodeURIComponent(datasetId)}`, {
        headers,
      });
      if (!res.ok) {
        return null;
      }
      const data = await res.json();
      const flows: NetworkFlowRecord[] = (data.flows || []).map((f: any) => ({
        id: f.id,
        flowId: f.flow_id || f.id,
        timestamp: f.timestamp,
        srcIp: f.src_ip,
        srcPort: f.src_port,
        dstIp: f.dst_ip,
        dstPort: f.dst_port,
        protocol: f.protocol,
        flowDurationMs: f.flow_duration_ms,
        totFwdPkts: f.tot_fwd_pkts,
        totBwdPkts: f.tot_bwd_pkts ?? null,
        pktLenMean: f.pkt_len_mean ?? null,
        flowBytsPerSec: f.flow_byts_per_sec ?? null,
        keyFeatures: f.key_features ?? null,
        flags: f.flags,
        predictedClass: f.predicted_class,
        actualLabel: f.actual_label,
        status: f.status || 'classified',
        isMalicious: f.is_malicious,
        confidence: f.confidence,
        riskLevel: f.risk_level,
        modelVersion: f.model_version,
        isSyntheticSample: false,
      }));

      return {
        total_flows: data.total_flows,
        malicious_count: data.malicious_count,
        benign_count: data.benign_count,
        failed_count: data.failed_count || 0,
        flows,
        model_version: data.model_version,
        dataset_id: data.dataset_id,
        dataset_name: data.dataset_name,
        per_class_counts: data.per_class_counts || {},
        per_class_percentages: data.per_class_percentages || {},
        evaluation: data.evaluation,
        forced_compatibility: data.forced_compatibility || false,
        imputed_features: data.imputed_features || [],
        compatibility_warning: data.compatibility_warning || null,
      };
    } catch {
      return null;
    }
  }

  /**
   * Retrieves active dataset evaluation metrics computed from model inference on ground-truth records.
   */
  async getDatasetEvaluation(datasetId: string): Promise<any | null> {
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${this.baseUrl}/api/v1/evaluation/dataset/${encodeURIComponent(datasetId)}`, {
        headers,
      });
      if (res.ok) {
        return await res.json();
      }
      return null;
    } catch (e) {
      console.warn('Failed to fetch dataset evaluation:', e);
      return null;
    }
  }

  /**
   * Uploads a labelled CSV dataset directly for empirical evaluation against FLMLP (mlp_model.pth).
   * Computes genuine Accuracy, Macro/Weighted F1, FPR, FNR, 7x7 Confusion Matrix and classification report.
   */
  async evaluateCsv(file: File, modelId?: string): Promise<any> {
    const headers = await getAuthHeaders();
    const formData = new FormData();
    formData.append('file', file);
    if (modelId) {
      formData.append('model_id', modelId);
    }

    const res = await fetch(`${this.baseUrl}/api/v1/evaluation/evaluate-csv`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Evaluation failed with status ${res.status}`);
    }

    return res.json();
  }

  /**
   * Retrieves history of persisted model evaluation runs.
   */
  async getEvaluationHistory(): Promise<any[]> {
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${this.baseUrl}/api/v1/evaluation/history`, {
        headers,
      });
      if (res.ok) {
        return await res.json();
      }
      return [];
    } catch (e) {
      console.warn('Failed to fetch evaluation history:', e);
      return [];
    }
  }

  /**
   * Retrieves a specific evaluation run by ID.
   */
  async getEvaluationById(evalId: string): Promise<any> {
    const headers = await getAuthHeaders();
    const res = await fetch(`${this.baseUrl}/api/v1/evaluation/run/${encodeURIComponent(evalId)}`, {
      headers,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Evaluation '${evalId}' not found`);
    }
    return res.json();
  }

  /**
   * Returns direct download URL for evaluation report export.
   */
  getEvaluationReportUrl(evalId: string, format: 'json' | 'markdown' | 'csv' = 'json'): string {
    return `${this.baseUrl}/api/v1/evaluation/report/${encodeURIComponent(evalId)}?format=${format}`;
  }

  /**
   * Retrieves Federated Learning operational status.
   */
  async getFLStatus(): Promise<{
    status: string;
    active_job_id?: string | null;
    available_scenarios: string[];
    clients_count: number;
    total_completed_runs: number;
    algorithm: string;
    model_architecture: string;
  }> {
    const res = await fetch(`${this.baseUrl}/api/v1/federated/status`);
    if (!res.ok) {
      throw new Error(`Failed to query FL status: ${res.statusText}`);
    }
    return res.json();
  }

  /**
   * Launches a new Federated Learning training job.
   */
  async startFLTraining(config: FLTrainingConfig): Promise<{
    job_id: string;
    status: string;
    scenario: string;
    current_round: number;
    total_rounds: number;
    started_at: string;
  }> {
    const headers = (await getAuthHeaders()) as Record<string, string>;
    headers['Content-Type'] = 'application/json';
    const res = await fetch(`${this.baseUrl}/api/v1/federated/train`, {
      method: 'POST',
      headers,
      body: JSON.stringify(config),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Failed to start FL training: ${res.status}`);
    }

    return res.json();
  }

  /**
   * Polls telemetry of an active FL training job.
   */
  async getFLJobStatus(jobId: string): Promise<{
    job_id: string;
    status: string;
    scenario: string;
    current_round: number;
    total_rounds: number;
    history: any[];
    latest_metrics?: any;
    started_at: string;
    completed_at?: string;
    error?: string;
  }> {
    const res = await fetch(`${this.baseUrl}/api/v1/federated/jobs/${jobId}`);
    if (!res.ok) {
      throw new Error(`Failed to query job ${jobId}: ${res.statusText}`);
    }
    return res.json();
  }

  /**
   * Retrieves active federated experiment or genuine completed run.
   */
  /**
   * Loads one FL run as an FLExperiment. With no runId, the saved A_iid benchmark run is used.
   * Every number comes from the backend; missing values stay undefined (never invented).
   */
  async getActiveExperiment(runId?: string): Promise<{ data: FLExperiment; isBackend: boolean }> {
    try {
      let targetId = runId;
      if (!targetId) {
        const runsRes = await fetch(`${this.baseUrl}/api/v1/federated/runs`);
        if (runsRes.ok) {
          const runs = await runsRes.json();
          const primary = runs.find((r: any) => !r.is_centralized && r.run_id.startsWith('A_iid')) || runs[0];
          targetId = primary?.run_id;
        }
      }
      if (targetId) {
        const detailRes = await fetch(`${this.baseUrl}/api/v1/federated/runs/${encodeURIComponent(targetId)}`);
        if (detailRes.ok) {
          return { data: mapFLRunDetail(await detailRes.json()), isBackend: true };
        }
      }
    } catch (e) {
      console.warn('Backend /api/v1/federated/ unreachable:', e);
    }

    return {
      data: null as any,
      isBackend: false,
    };
  }

  /**
   * Retrieves genuine empirical evaluation experiments from FastAPI backend.
   */
  async getEvaluationExperiments(): Promise<{ data: EvaluationExperiment[]; isBackend: boolean }> {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/evaluation/benchmarks`);
      if (res.ok) {
        const benchmarks = await res.json();
        if (Array.isArray(benchmarks) && benchmarks.length > 0) {
          return { data: benchmarks, isBackend: true };
        }
      }
    } catch (e) {
      console.warn('Backend /api/v1/evaluation/benchmarks unreachable:', e);
    }

    return {
      data: [],
      isBackend: false,
    };
  }

  async getDatasets(): Promise<{ data: Dataset[]; isBackend: boolean }> {
    return {
      data: storage.getDatasets(false),
      isBackend: false,
    };
  }

  async getDetectionFlows(): Promise<{ data: NetworkFlowRecord[]; isBackend: boolean }> {
    return {
      data: [],
      isBackend: false,
    };
  }

  /**
   * Retrieves durable audit logs from FastAPI backend SQLite ledger.
   */
  async getAuditLogs(params?: {
    query?: string;
    severity?: string;
    category?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    logs: LogEntry[];
    total: number;
    limit: number;
    offset: number;
    categories_count: Record<string, number>;
    severities_count: Record<string, number>;
  }> {
    const url = new URL(`${this.baseUrl}/api/v1/logs`);
    if (params?.query) url.searchParams.set('query', params.query);
    if (params?.severity && params.severity !== 'ALL') url.searchParams.set('severity', params.severity);
    if (params?.category && params.category !== 'ALL') url.searchParams.set('category', params.category);
    if (params?.limit) url.searchParams.set('limit', String(params.limit));
    if (params?.offset) url.searchParams.set('offset', String(params.offset));

    const res = await fetch(url.toString());
    if (!res.ok) {
      throw new Error(`Failed to fetch audit logs: ${res.statusText}`);
    }
    return res.json();
  }

  /**
   * Retrieves live platform architecture, capabilities, and true asset counts.
   */
  async getSystemSummary(): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/v1/system/summary`);
    if (!res.ok) {
      throw new Error(`Failed to fetch system summary: ${res.statusText}`);
    }
    return res.json();
  }
}

export const apiClient = new ApiClient();


/** Maps a backend FL run (saved benchmark or a job started from the app) to the UI model. */
export function mapFLRunDetail(detail: any): FLExperiment {
  const src = detail.data_source || {};
  const isUploaded = src.kind === 'uploaded_dataset';
  const scenario = detail.scenario || 'A_iid';
  const clients = detail.clients || [];
  const roundsHistory: FLRoundSummary[] = (detail.round_history || []).map((rh: any) => ({
    round: rh.round,
    globalLoss: rh.fed_val_loss ?? rh.train_loss ?? 0,
    globalAccuracy: rh.fed_val_acc ?? rh.train_acc ?? 0,
    globalF1: rh.fed_val_macroF1 ?? 0,
    activeClientsCount: rh.n_train_clients ?? clients.length,
    durationSeconds: rh.max_client_secs ?? rh.duration_seconds ?? 0,
    // Per-client values are only recorded for the final round
    clientMetrics: clients.map((c: any) => ({
      clientId: c.id,
      clientName: c.name,
      localLoss: c.localLoss,
      localAccuracy: c.localAccuracy,
      localF1: c.localF1,
      sampleWeight: c.sampleWeight,
      samplesUsed: c.sampleCount,
    })),
    aggregatedAt: detail.completed_at || '',
  }));
  const partition = isUploaded ? src.partition : (scenario.includes('dir') ? 'dirichlet' : 'iid');
  return {
    id: detail.run_id,
    name: `FedAvg on ${src.dataset_name || 'CIC-IoT-2023 benchmark'} (${scenario})`,
    description: isUploaded
      ? `Trained on your uploaded dataset: ${src.rows_used ?? '?'} cleaned rows, ${src.test_rows ?? '?'} held out for testing.`
      : 'Trained on the built-in CIC-IoT-2023 benchmark cache and its fixed client split files.',
    datasetName: src.dataset_name || 'CIC-IoT-2023 benchmark',
    dataSource: src,
    roundsTotal: detail.rounds_total || roundsHistory.length,
    currentRound: detail.current_round || roundsHistory.length,
    status: detail.status === 'completed' ? 'completed' : 'running',
    partitionStrategy: partition === 'dirichlet' ? 'non_iid_dirichlet' : 'iid',
    dirichletAlpha: scenario.includes('dir0.1') ? 0.1 : (scenario.includes('dir0.5') ? 0.5 : undefined),
    clientParticipationFraction: 1.0,
    learningRate: detail.config?.lr ?? 0.001,
    batchSize: detail.config?.batch_size ?? 512,
    randomSeed: detail.config?.seed ?? 42,
    clients: clients.map((c: any) => ({
      id: c.id,
      name: c.name,
      clientType: c.clientType || 'smart_home',
      deviceTypes: [],
      simulatedEnvironment: 'Simulated client (in-process)',
      sampleCount: c.sampleCount ?? 0,
      dataDistribution: partition === 'dirichlet' ? 'non_iid' : 'iid',
      status: 'synced',
      localEpochs: detail.config?.local_epochs ?? 1,
      sampleWeight: c.sampleWeight ?? 0,
      localLoss: c.localLoss,
      localAccuracy: c.localAccuracy,
      localF1: c.localF1,
    })),
    roundsHistory,
    globalModelVersion: 'FedAvg-FLMLP-v1.0',
    createdAt: detail.completed_at || detail.started_at || '',
    isDemonstrationRun: false,
  };
}
