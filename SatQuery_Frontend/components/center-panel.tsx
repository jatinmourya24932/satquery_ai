'use client';

import dynamic from 'next/dynamic';
import {
  Layers,
  Eye,
  Search,
  Mic,
  Send,
  ZoomIn,
  Image as ImageIcon,
  X,
  Loader2,
  Bot,
  User,
  ShieldCheck,
  Map as MapIcon,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import type { AnalyzeResult } from '@/lib/api';
import type { ChatMessage } from '@/app/page';

const SatelliteMap = dynamic(
  () => import('./satellite-map').then((mod) => mod.SatelliteMap),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full items-center justify-center bg-slate-100">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-600/20 border-t-blue-600" />
          <span className="text-xs text-slate-500">Loading satellite imagery...</span>
        </div>
      </div>
    ),
  }
);

const layerPills = [
  { id: 'true-color', label: 'True Color', color: 'text-cyan-600', bg: 'bg-cyan-500' },
  { id: 'ndvi', label: 'NDVI Vegetation', color: 'text-emerald-700', bg: 'bg-emerald-600' },
  { id: 'lulc', label: 'LULC', color: 'text-amber-600', bg: 'bg-amber-500' },
  { id: 'water', label: 'Water Mask', color: 'text-blue-600', bg: 'bg-blue-500' },
  { id: 'urban', label: 'Urban', color: 'text-orange-600', bg: 'bg-orange-500' },
  { id: 'change', label: 'Change Detection', color: 'text-pink-600', bg: 'bg-pink-500' },
];

const legendItems: Record<string, { label: string; color: string }[]> = {
  'True Color': [
    { label: 'Vegetation', color: '#22c55e' },
    { label: 'Water Bodies', color: '#3b82f6' },
    { label: 'Urban / Built-up', color: '#94a3b8' },
    { label: 'Bare Soil', color: '#d4a373' },
  ],
  'NDVI Vegetation': [
    { label: 'Dense Veg (0.6-1.0)', color: '#166534' },
    { label: 'Moderate (0.3-0.6)', color: '#4ade80' },
    { label: 'Sparse (0.1-0.3)', color: '#facc15' },
    { label: 'No Veg (< 0.1)', color: '#78350f' },
  ],
  LULC: [
    { label: 'Agricultural', color: '#84cc16' },
    { label: 'Forest', color: '#15803d' },
    { label: 'Water', color: '#0284c7' },
    { label: 'Built-up', color: '#a8a29e' },
  ],
  'Water Mask': [
    { label: 'Water Surface', color: '#0ea5e9' },
    { label: 'Non-Water', color: '#1e293b' },
  ],
  Urban: [
    { label: 'High Density', color: '#f97316' },
    { label: 'Medium Density', color: '#fb923c' },
    { label: 'Low Density', color: '#fcd34d' },
    { label: 'Non-Urban', color: '#334155' },
  ],
  'Change Detection': [
    { label: 'Gain / New', color: '#22d3ee' },
    { label: 'Loss / Removed', color: '#f43f5e' },
    { label: 'Stable', color: '#64748b' },
  ],
};

const presetQueries = [
  'Describe the land-cover and major objects visible in this image.',
  'What is the dominant land use in this scene?',
  'Identify built-up and water regions.',
  'Locate and highlight the water bodies in this image.',
  'Highlight the built-up / urban areas.',
  'Has the built-up area increased, decreased, or remained unchanged?',
];

interface CenterPanelProps {
  onQuerySubmit: (query: string, images?: File[]) => void;
  showPolygon: boolean;
  isAnalyzing: boolean;
  result?: AnalyzeResult | null;
  error?: string | null;
  messages: ChatMessage[];
}

export function CenterPanel({
  onQuerySubmit,
  showPolygon,
  isAnalyzing,
  result,
  error,
  messages,
}: CenterPanelProps) {
  const [activeLayer, setActiveLayer] = useState('True Color');
  const [searchValue, setSearchValue] = useState('');
  const [showLegend, setShowLegend] = useState(true);
  const [showPresets, setShowPresets] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [forceMap, setForceMap] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const latestEvidence = [...messages]
    .reverse()
    .find((m) => m.role === 'assistant' && (m.evidenceUrl || m.changeMaskUrl));

  const showEvidencePanel = !forceMap && !!latestEvidence?.evidenceUrl && !isAnalyzing;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAnalyzing]);

  useEffect(() => {
    if (isAnalyzing) setForceMap(false);
  }, [isAnalyzing]);

  const handleSend = () => {
    if (!searchValue.trim() || isAnalyzing) return;
    const q = searchValue;
    setSearchValue('');
    onQuerySubmit(q, selectedFiles.length > 0 ? selectedFiles : undefined);
  };

  const handlePreset = (preset: string) => {
    setSearchValue(preset);
    setShowPresets(false);
    if (!isAnalyzing) {
      onQuerySubmit(preset, selectedFiles.length > 0 ? selectedFiles : undefined);
      setSearchValue('');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    setSelectedFiles(Array.from(files).slice(0, 2));
  };

  const clearFiles = () => {
    setSelectedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="relative flex h-full flex-1 flex-col min-w-0">
      {/* Top bar */}
      <div className="z-[500] flex items-center gap-2 border-b border-slate-200 bg-white px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-500">
          <Layers className="h-3.5 w-3.5 text-blue-700" />
          Layers
        </div>
        <div className="flex flex-1 flex-wrap items-center gap-1.5 overflow-x-auto">
          {layerPills.map((pill) => (
            <button
              key={pill.id}
              onClick={() => setActiveLayer(pill.label)}
              className={cn(
                'whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-200',
                activeLayer === pill.label
                  ? 'bg-blue-600 text-white shadow-[0_2px_8px_rgba(30,64,175,0.2)]'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900'
              )}
            >
              <span className={cn('mr-1.5 inline-block h-1.5 w-1.5 rounded-full', pill.bg)} />
              {pill.label}
            </button>
          ))}
        </div>

        {latestEvidence?.evidenceUrl && (
          <button
            onClick={() => setForceMap((v) => !v)}
            className={cn(
              'flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors',
              forceMap
                ? 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
                : 'bg-blue-50 text-blue-700 ring-1 ring-blue-200'
            )}
          >
            {forceMap ? (
              <>
                <ShieldCheck className="h-3.5 w-3.5" />
                Evidence
              </>
            ) : (
              <>
                <MapIcon className="h-3.5 w-3.5" />
                Map
              </>
            )}
          </button>
        )}

        <button
          onClick={() => setShowLegend(!showLegend)}
          className={cn(
            'flex h-8 w-8 items-center justify-center rounded-lg transition-colors',
            showLegend ? 'bg-blue-600/10 text-blue-700' : 'text-slate-500 hover:bg-slate-100'
          )}
        >
          <Eye className="h-4 w-4" />
        </button>
      </div>

      {/* TOP: Evidence OR Map */}
      <div className="relative h-[48%] min-h-[220px] shrink-0 overflow-hidden border-b border-slate-200 bg-slate-900">
        {showEvidencePanel ? (
          <div className="relative flex h-full w-full flex-col">
            <div className="absolute left-3 top-3 z-10 flex items-center gap-2 rounded-lg border border-emerald-400/40 bg-emerald-950/80 px-3 py-1.5 backdrop-blur-md">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-300">
                Visual Evidence / Proof
              </span>
              {result?.bbox && (
                <span className="rounded bg-red-500/90 px-1.5 py-0.5 text-[9px] font-bold text-white">
                  RED BOX = DETECTED REGION
                </span>
              )}
            </div>

            <div className="flex h-full w-full items-center justify-center bg-slate-950 p-2">
              <img
                src={latestEvidence!.evidenceUrl}
                alt="Visual evidence with highlights"
                className="max-h-full max-w-full rounded-lg object-contain shadow-2xl ring-1 ring-white/10"
              />
            </div>

            {latestEvidence?.changeMaskUrl && (
              <div className="absolute bottom-3 right-3 z-10 w-36 overflow-hidden rounded-lg border border-pink-400/40 bg-slate-950/90 shadow-lg">
                <div className="bg-pink-600/80 px-2 py-0.5 text-[9px] font-bold uppercase text-white">
                  Change Mask
                </div>
                <img
                  src={latestEvidence.changeMaskUrl}
                  alt="Change mask"
                  className="h-24 w-full object-cover"
                />
              </div>
            )}
          </div>
        ) : (
          <>
            <SatelliteMap showPolygon={showPolygon} layerType={activeLayer} />

            {isAnalyzing && (
              <div className="pointer-events-none absolute inset-0 z-[400] overflow-hidden">
                <div className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-blue-600 to-transparent animate-scan" />
              </div>
            )}

            {showLegend && (
              <div className="absolute right-3 top-3 z-[500] w-48 rounded-xl border border-white/60 bg-white/82 p-3 backdrop-blur-xl shadow-lg">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-blue-700">
                    Layer Legend
                  </span>
                  <span className="text-[9px] text-slate-400">{activeLayer}</span>
                </div>
                <div className="space-y-1">
                  {(legendItems[activeLayer] || legendItems['True Color']).map((item) => (
                    <div key={item.label} className="flex items-center gap-2">
                      <div
                        className="h-2 w-2 rounded-sm ring-1 ring-slate-200"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-[10px] text-slate-700">{item.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="absolute left-3 top-3 z-[500] flex items-center gap-2 rounded-lg border border-white/60 bg-white/82 px-2.5 py-1 backdrop-blur-xl shadow-sm">
              <div className="h-2 w-2 rounded-full bg-emerald-600 animate-pulse-live" />
              <span className="text-[11px] font-medium text-slate-900">Khandwa, MP</span>
            </div>

            <div className="absolute bottom-3 right-3 z-[500] flex items-center gap-1.5 rounded-lg border border-white/60 bg-white/82 px-2 py-1 backdrop-blur-xl shadow-sm">
              <ZoomIn className="h-3 w-3 text-blue-700" />
              <span className="text-[10px] text-slate-600">Zoom 11</span>
            </div>
          </>
        )}
      </div>

      {/* Chat */}
      <div className="flex min-h-0 flex-1 flex-col bg-slate-50">
        <div className="flex items-center gap-2 border-b border-slate-200 bg-white px-4 py-2">
          <Bot className="h-4 w-4 text-blue-700" />
          <span className="text-xs font-bold uppercase tracking-widest text-slate-600">
            Analysis Chat
          </span>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
          {messages.length === 0 && !isAnalyzing && (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-slate-400">
              <Bot className="h-8 w-8 opacity-40" />
              <p className="text-sm">Ask a question about the satellite imagery</p>
              <p className="text-[11px]">
                Tip: “locate / highlight / where is” → red boxes on evidence image
              </p>
            </div>
          )}

          {messages.map((m) => (
            <div
              key={m.id}
              className={cn('flex gap-2', m.role === 'user' ? 'justify-end' : 'justify-start')}
            >
              {m.role === 'assistant' && (
                <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white">
                  <Bot className="h-3.5 w-3.5" />
                </div>
              )}

              <div
                className={cn(
                  'max-w-[85%] rounded-2xl px-3.5 py-2.5 shadow-sm',
                  m.role === 'user'
                    ? 'rounded-br-md bg-blue-600 text-white'
                    : m.error
                    ? 'rounded-bl-md border border-red-200 bg-red-50 text-red-800'
                    : 'rounded-bl-md border border-slate-200 bg-white text-slate-800'
                )}
              >
                {m.previewUrls && m.previewUrls.length > 0 && (
                  <div className="mb-2 flex flex-wrap gap-1.5">
                    {m.previewUrls.map((url, i) => (
                      <img
                        key={i}
                        src={url}
                        alt={`upload-${i}`}
                        className="h-14 w-14 rounded-lg object-cover ring-1 ring-white/40"
                      />
                    ))}
                  </div>
                )}

                <p className="whitespace-pre-wrap text-sm leading-relaxed">{m.text}</p>

                {m.role === 'assistant' && !m.error && (m.confidence != null || m.models) && (
                  <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-2 text-[10px] text-slate-500">
                    {m.confidence != null && (
                      <span className="rounded-full bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700 ring-1 ring-emerald-200">
                        {Math.round(m.confidence * 100)}% conf
                      </span>
                    )}
                    {m.models?.map((model) => (
                      <code key={model} className="rounded bg-slate-100 px-1.5 py-0.5">
                        {model}
                      </code>
                    ))}
                    {m.evidenceUrl && (
                      <span className="rounded-full bg-red-50 px-2 py-0.5 font-medium text-red-700 ring-1 ring-red-200">
                        proof on image ↑
                      </span>
                    )}
                  </div>
                )}
              </div>

              {m.role === 'user' && (
                <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-200 text-slate-600">
                  <User className="h-3.5 w-3.5" />
                </div>
              )}
            </div>
          ))}

          {isAnalyzing && (
            <div className="flex justify-start gap-2">
              <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white">
                <Bot className="h-3.5 w-3.5" />
              </div>
              <div className="rounded-2xl rounded-bl-md border border-slate-200 bg-white px-4 py-3">
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                  Agent is analyzing…
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-200 bg-white px-4 py-3">
          {selectedFiles.length > 0 && (
            <div className="mb-2 flex flex-wrap items-center gap-2">
              {selectedFiles.map((f, i) => (
                <span
                  key={`${f.name}-${i}`}
                  className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-2.5 py-1 text-[11px] font-medium text-blue-700 ring-1 ring-blue-200"
                >
                  <ImageIcon className="h-3 w-3" />
                  {f.name.length > 20 ? f.name.slice(0, 17) + '…' : f.name}
                </span>
              ))}
              <button onClick={clearFiles} className="rounded-full p-1 text-slate-400 hover:bg-slate-100">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          <div className="relative flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm focus-within:border-blue-600/40">
            <Search className="h-4 w-4 shrink-0 text-slate-400" />
            <input
              type="text"
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              onFocus={() => setShowPresets(true)}
              onBlur={() => setTimeout(() => setShowPresets(false), 200)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask about the satellite data…"
              disabled={isAnalyzing}
              className="flex-1 bg-transparent text-sm focus:outline-none disabled:opacity-60"
            />

            {showPresets && (
              <div className="absolute bottom-full left-0 right-0 z-50 mb-2 rounded-xl border border-slate-200 bg-white p-2 shadow-xl">
                <div className="mb-1 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Example Queries
                </div>
                {presetQueries.map((preset) => (
                  <button
                    key={preset}
                    onClick={() => handlePreset(preset)}
                    className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-slate-600 hover:bg-slate-100"
                  >
                    <Search className="h-3 w-3 text-blue-600" />
                    {preset}
                  </button>
                ))}
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.tif,.tiff"
              multiple
              className="hidden"
              onChange={handleFileChange}
            />

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 hover:bg-slate-100 disabled:opacity-40"
            >
              <ImageIcon className="h-4 w-4" />
            </button>

            <button className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 opacity-40" disabled>
              <Mic className="h-4 w-4" />
            </button>

            <button
              onClick={handleSend}
              disabled={!searchValue.trim() || isAnalyzing}
              className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-emerald-700 to-emerald-600 px-3.5 py-1.5 text-sm font-semibold text-white disabled:opacity-40"
            >
              {isAnalyzing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
              Send
            </button>
          </div>

          <p className="mt-1.5 text-[10px] text-slate-400">
            Analysis ke baad map → evidence image. Toggle se map wapas aa sakta hai.
          </p>
        </div>
      </div>
    </div>
  );
}