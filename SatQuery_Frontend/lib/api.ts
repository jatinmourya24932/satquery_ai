const API_BASE =
  process.env.NEXT_PUBLIC_SATQUERY_API_URL || 'http://localhost:8000';

export interface ExecutionStep {
  step: string;
  timestamp?: string;
  details?: Record<string, unknown>;
}

export interface ExecutionSummary {
  total_steps?: number;
  duration_sec?: number;
  steps?: ExecutionStep[];
}

export interface QuantitativeSummary {
  metrics?: Record<string, number | string | number[]>;
  area_analysis?: Record<string, unknown>;
  num_specialist_outputs?: number;
}

export interface AnalyzeResult {
  success: boolean;
  query?: string;
  input_type?: string;
  modalities?: string[];
  primary_task?: string;
  plan?: string;
  answer?: string;
  confidence?: number;
  bbox?: [number, number, number, number] | null;
  models_used?: string[];
  validation_summary?: string;
  quantitative_summary?: QuantitativeSummary;
  geospatial_metadata?: Record<string, unknown>[];
  execution_summary?: ExecutionSummary;
  visual_evidence_b64?: string | null;
  change_mask_b64?: string | null;
  error?: string;
}

export interface AnalyzeOptions {
  query: string;
  images?: File[];
  useSamples?: string;
  modalityHints?: string;
}

export async function analyzeQuery(
  options: AnalyzeOptions
): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append('query', options.query);

  if (options.images && options.images.length > 0) {
    options.images.forEach((file) => form.append('images', file));
  }
  if (options.useSamples) form.append('use_samples', options.useSamples);
  if (options.modalityHints) form.append('modality_hints', options.modalityHints);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail || err.error || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export { API_BASE };