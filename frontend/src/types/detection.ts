export type ProtocolType = 'TCP' | 'UDP' | 'ICMP' | 'OTHER';
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

export interface NetworkFlowRecord {
  id: string;
  flowId?: string;
  timestamp: string;
  // Endpoint / derived fields are present only when the source data contains them
  // (CIC IoT 2023 flow features have no IPs or ports). Never fill them with placeholders.
  srcIp?: string | null;
  srcPort?: number | null;
  dstIp?: string | null;
  dstPort?: number | null;
  protocol?: ProtocolType | string | null;
  flowDurationMs?: number | null;
  totFwdPkts?: number | null;
  totBwdPkts?: number | null;
  pktLenMean?: number | null;
  flowBytsPerSec?: number | null;
  flags?: string | null;
  /** Raw values of canonical model input features for this flow, straight from the data. */
  keyFeatures?: Record<string, number> | null;
  predictedClass: string;
  actualLabel?: string;
  status?: 'classified' | 'failed' | 'pending' | 'rejected';
  isMalicious: boolean;
  confidence: number;
  riskLevel: RiskLevel;
  modelVersion: string;
  isSyntheticSample?: boolean;
}

export interface BatchClassificationResult {
  total_flows: number;
  malicious_count: number;
  benign_count: number;
  failed_count: number;
  flows: NetworkFlowRecord[];
  model_version: string;
  model_id?: string;
  dataset_id?: string;
  dataset_name?: string;
  per_class_counts: Record<string, number>;
  per_class_percentages: Record<string, number>;
  evaluation?: any;
}

export interface DetectionFilter {
  searchQuery: string;
  classFilter: string; // 'all' | 'malicious_only' | 'benign_only' | specific class
  protocolFilter: string;
  minConfidence: number;
  riskFilter: string;
}

