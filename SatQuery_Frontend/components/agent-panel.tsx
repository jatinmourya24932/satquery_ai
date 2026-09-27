'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  CheckCircle2,
  Circle,
  Loader2,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Leaf,
  Droplets,
  Building2,
  Brain,
  AlertCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type { AnalyzeResult, ExecutionStep } from '@/lib/api';

interface AgentTrace {
  id: number;
  label: string;
  timestamp?: string;
}

const FALLBACK_STEPS: AgentTrace[] = [
  { id: 1, label: 'Query Received' },
  { id: 2, label: 'Validation & Preprocessing' },
  { id: 3, label: 'Task Classification' },
  { id: 4, label: 'Model Selection' },
  { id: 5, label: 'AI Analysis' },
  { id: 6, label: 'Fusion & Results' },
];

function stepLabel(step: ExecutionStep): string {
  const map: Record<string, string> = {
    start: 'Query Received',
    validation: 'Validation & Preprocessing',
    task_classification: 'Task Classification',
    classify: 'Task Classification',
    select_model: 'Model Selection',
    execution: 'AI Analysis',
    fusion: 'Fusion & Results',
    quantitative_summary: 'Quantitative Summary',
    geospatial_processing: 'Geospatial Processing',
    complete: 'Results Generated',
  };
  return map[step.step] || step.step.replace(/_/g, ' ');
}

interface Finding {
  id: string;
  title: string;
  value: string;
  trend: 'up' | 'down' | 'stable';
  trendLabel: string;
  icon: typeof Leaf;
  color: string;
  bgColor: string;
  ringColor: string;
}

function buildFindings(result: AnalyzeResult | null): Finding[] {
  if (!result?.success) return [];

  const metrics = result.quantitative_summary?.metrics || {};
  const findings: Finding[] = [];

  const changePct = metrics.change_ratio_pct;
  if (typeof changePct === 'number') {
    findings.push({
      id: 'change',
      title: 'Change Ratio',
      value: `${changePct > 0 ? '+' : ''}${changePct}%`,
      trend: changePct > 1 ? 'up' : changePct < -1 ? 'down' : 'stable',
      trendLabel: 'Detected change area',
      icon: Building2,
      color: 'text-orange-700',
      bgColor: 'bg-orange-50',
      ringColor: 'ring-orange-200',
    });
  }

  const conf = result.confidence;
  if (typeof conf === 'number') {
    findings.push({
      id: 'conf',
      title: 'Model Confidence',
      value: `${Math.round(conf * 100)}%`,
      trend: conf >= 0.7 ? 'up' : conf >= 0.4 ? 'stable' : 'down',
      trendLabel: result.primary_task || 'Analysis',
      icon: Leaf,
      color: 'text-emerald-700',
      bgColor: 'bg-emerald-50',
      ringColor: 'ring-emerald-200',
    });
  }

  const area = result.quantitative_summary?.area_analysis as
    | { changed_area_ha?: number; changed_area_km2?: number }
    | undefined;
  if (area?.changed_area_ha != null) {
    findings.push({
      id: 'area',
      title: 'Changed Area',
      value: `${Number(area.changed_area_ha).toFixed(2)} ha`,
      trend: 'stable',
      trendLabel: area.changed_area_km2
        ? `${Number(area.changed_area_km2).toFixed(3)} km²`
        : 'From geo stats',
      icon: Droplets,
      color: 'text-blue-700',
      bgColor: 'bg-blue-50',
      ringColor: 'ring-blue-200',
    });
  }

  if (findings.length === 0 && result.models_used?.length) {
    findings.push({
      id: 'model',
      title: 'Models Used',
      value: result.models_used[0],
      trend: 'stable',
      trendLabel: `${result.models_used.length} specialist(s)`,
      icon: Brain,
      color: 'text-blue-700',
      bgColor: 'bg-blue-50',
      ringColor: 'ring-blue-200',
    });
  }

  return findings.slice(0, 3);
}

interface AgentPanelProps {
  analysisTrigger: number;
  isAnalyzing: boolean;
  result?: AnalyzeResult | null;
}

export function AgentPanel({
  analysisTrigger,
  isAnalyzing,
  result,
}: AgentPanelProps) {
  const [completedSteps, setCompletedSteps] = useState(0);
  const [showFindings, setShowFindings] = useState(false);
  const [confidence, setConfidence] = useState(0);

  const liveSteps: AgentTrace[] = useMemo(() => {
    const steps = result?.execution_summary?.steps;
    if (steps && steps.length > 0) {
      return steps.map((s, i) => ({
        id: i + 1,
        label: stepLabel(s),
        timestamp: s.timestamp
          ? new Date(s.timestamp).toLocaleTimeString('en-IN', {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            })
          : undefined,
      }));
    }
    return FALLBACK_STEPS;
  }, [result]);

  const findings = useMemo(() => buildFindings(result ?? null), [result]);

  useEffect(() => {
    if (analysisTrigger === 0) return;

    setCompletedSteps(0);
    setShowFindings(false);
    setConfidence(0);

    if (!isAnalyzing && result) {
      setCompletedSteps(liveSteps.length);
      setShowFindings(true);
      const target = Math.round((result.confidence ?? 0) * 100) || 0;
      let current = 0;
      const iv = setInterval(() => {
        current += 4;
        if (current >= target) {
          current = target;
          clearInterval(iv);
        }
        setConfidence(current);
      }, 25);
      return () => clearInterval(iv);
    }

    const timeouts: ReturnType<typeof setTimeout>[] = [];
    const n = FALLBACK_STEPS.length;
    for (let i = 0; i < n - 1; i++) {
      timeouts.push(setTimeout(() => setCompletedSteps(i + 1), (i + 1) * 700));
    }
    return () => timeouts.forEach(clearTimeout);
  }, [analysisTrigger, isAnalyzing, result, liveSteps.length]);

  const circumference = 2 * Math.PI * 52;
  const dashOffset = circumference - (confidence / 100) * circumference;

  return (
    <aside className="flex h-full w-[380px] flex-col border-l border-slate-200 bg-slate-100 overflow-y-auto">
      <div className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4 shadow-sm">
        <div className="flex items-center gap-2.5">
          <Brain className="h-5 w-5 text-blue-700" />
          <h2 className="text-sm font-bold tracking-wide text-slate-900">AGENT ANALYSIS</h2>
        </div>
        <div
          className={cn(
            'flex items-center gap-1.5 rounded-full px-2.5 py-1 ring-1',
            isAnalyzing
              ? 'bg-red-50 ring-red-200'
              : result?.success
              ? 'bg-emerald-50 ring-emerald-200'
              : 'bg-slate-50 ring-slate-200'
          )}
        >
          <div
            className={cn(
              'h-2 w-2 rounded-full',
              isAnalyzing
                ? 'bg-red-500 animate-pulse-live'
                : result?.success
                ? 'bg-emerald-500'
                : 'bg-slate-400'
            )}
          />
          <span
            className={cn(
              'text-[10px] font-bold uppercase tracking-widest',
              isAnalyzing
                ? 'text-red-600'
                : result?.success
                ? 'text-emerald-700'
                : 'text-slate-500'
            )}
          >
            {isAnalyzing ? 'LIVE' : result?.success ? 'DONE' : 'IDLE'}
          </span>
        </div>
      </div>

      {result?.plan && (
        <div className="border-b border-slate-200 bg-blue-50/80 px-5 py-2.5">
          <p className="text-[11px] text-blue-800 leading-relaxed">{result.plan}</p>
        </div>
      )}

      <div className="border-b border-slate-200 px-5 py-4">
        <div className="mb-3 flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-blue-700" />
          <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">
            Execution Trace
          </span>
        </div>

        <div className="space-y-0">
          {liveSteps.map((step, idx) => {
            const isCompleted = idx < completedSteps;
            const isCurrent = idx === completedSteps && isAnalyzing;
            const isPending = idx > completedSteps;

            return (
              <div
                key={`${step.id}-${step.label}`}
                className={cn(
                  'relative flex items-start gap-3 transition-opacity duration-300',
                  isPending && 'opacity-40'
                )}
              >
                {idx < liveSteps.length - 1 && (
                  <div
                    className={cn(
                      'absolute left-[11px] top-7 h-9 w-px',
                      isCompleted ? 'bg-emerald-600/40' : 'bg-slate-200'
                    )}
                  />
                )}

                <div
                  className={cn(
                    'relative z-10 mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full transition-all duration-300',
                    isCompleted && 'bg-emerald-100 ring-1 ring-emerald-300',
                    isCurrent && 'bg-blue-50 ring-1 ring-blue-300',
                    isPending && 'bg-slate-100 ring-1 ring-slate-200'
                  )}
                >
                  {isCompleted ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  ) : isCurrent ? (
                    <Loader2 className="h-3.5 w-3.5 text-blue-600 animate-spin" />
                  ) : (
                    <Circle className="h-3.5 w-3.5 text-slate-300" />
                  )}
                </div>

                <div className="flex-1 pb-4">
                  <div
                    className={cn(
                      'text-xs font-medium transition-colors',
                      isCompleted ? 'text-slate-900' : 'text-slate-500'
                    )}
                  >
                    Step {step.id}: {step.label}
                  </div>
                  {step.timestamp && isCompleted && (
                    <div className="text-[10px] text-slate-400">{step.timestamp}</div>
                  )}
                  {isCurrent && (
                    <div className="text-[10px] text-blue-600 animate-pulse">Processing...</div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="border-b border-slate-200 bg-white px-5 py-5">
        <div className="mb-3 flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-emerald-700" />
          <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">
            Confidence Score
          </span>
        </div>

        <div className="flex items-center justify-center">
          <div className="relative h-36 w-36">
            <svg className="h-full w-full -rotate-90" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#E2E8F0" strokeWidth="8" />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke="#059669"
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={dashOffset}
                className="transition-all duration-300"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-bold text-slate-900">{confidence}</span>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                percent
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 px-5 py-4">
        <div className="mb-3 flex items-center gap-2">
          <Leaf className="h-3.5 w-3.5 text-emerald-700" />
          <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">
            Key Findings
          </span>
        </div>

        {!showFindings && isAnalyzing ? (
          <div className="flex flex-col items-center gap-3 py-8 text-slate-400">
            <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
            <span className="text-xs">Running agent pipeline…</span>
          </div>
        ) : result && !result.success ? (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-4 w-4 shrink-0 text-red-600 mt-0.5" />
              <div>
                <div className="text-xs font-semibold text-red-800">Analysis failed</div>
                <p className="mt-1 text-[11px] text-red-700">{result.error || 'Unknown error'}</p>
              </div>
            </div>
          </div>
        ) : showFindings && findings.length > 0 ? (
          <div className="space-y-2.5">
            {findings.map((finding, idx) => {
              const Icon = finding.icon;
              const TrendIcon =
                finding.trend === 'up'
                  ? TrendingUp
                  : finding.trend === 'down'
                  ? TrendingDown
                  : Minus;

              return (
                <div
                  key={finding.id}
                  className={cn(
                    'animate-fade-in-up rounded-xl border border-slate-200 p-3.5 opacity-0 ring-1',
                    finding.bgColor,
                    finding.ringColor
                  )}
                  style={{ animationDelay: `${idx * 150}ms`, animationFillMode: 'forwards' }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2.5">
                      <div
                        className={cn(
                          'flex h-8 w-8 items-center justify-center rounded-lg bg-white shadow-sm',
                          finding.color
                        )}
                      >
                        <Icon className="h-4 w-4" />
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-slate-900">{finding.title}</div>
                        <div className="text-[10px] text-slate-500">{finding.trendLabel}</div>
                      </div>
                    </div>
                    <div className={cn('flex items-center gap-1', finding.color)}>
                      <TrendIcon className="h-4 w-4" />
                    </div>
                  </div>
                  <div className={cn('mt-2 text-lg font-bold', finding.color)}>{finding.value}</div>
                </div>
              );
            })}

            {result?.models_used && result.models_used.length > 0 && (
              <div className="mt-2 text-[10px] text-slate-500">
                Models:{' '}
                {result.models_used.map((m) => (
                  <code key={m} className="mr-1 rounded bg-slate-200 px-1 py-0.5">
                    {m}
                  </code>
                ))}
              </div>
            )}

            <div
              className="mt-3 flex items-center justify-center gap-2 rounded-lg bg-emerald-50 py-2.5 ring-1 ring-emerald-200 animate-fade-in-up opacity-0"
              style={{ animationDelay: '500ms', animationFillMode: 'forwards' }}
            >
              <div className="h-2 w-2 rounded-full bg-emerald-600 animate-pulse-live" />
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-700">
                Analysis Complete
              </span>
            </div>
          </div>
        ) : showFindings && result?.answer ? (
          <div className="rounded-xl border border-slate-200 bg-white p-3.5 text-sm text-slate-700">
            {result.answer}
          </div>
        ) : (
          <div className="py-6 text-center text-xs text-slate-400">
            Submit a query to start agent analysis
          </div>
        )}
      </div>
    </aside>
  );
}