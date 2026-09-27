'use client';

import { useState, useCallback } from 'react';
import { Sidebar } from '@/components/sidebar';
import { CenterPanel } from '@/components/center-panel';
import { AgentPanel } from '@/components/agent-panel';
import { analyzeQuery, type AnalyzeResult } from '@/lib/api';

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  previewUrls?: string[];
  evidenceUrl?: string;
  changeMaskUrl?: string;
  confidence?: number;
  models?: string[];
  error?: boolean;
};

export default function Home() {
  const [analysisTrigger, setAnalysisTrigger] = useState(0);
  const [showPolygon, setShowPolygon] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const handleQuerySubmit = useCallback(
    async (query: string, images?: File[]) => {
      if (!query.trim()) return;

      const previewUrls =
        images && images.length > 0
          ? images.map((f) => URL.createObjectURL(f))
          : undefined;

      setMessages((prev) => [
        ...prev,
        {
          id: `u-${Date.now()}`,
          role: 'user',
          text: query.trim(),
          previewUrls,
        },
      ]);

      setIsAnalyzing(true);
      setShowPolygon(false);
      setError(null);
      setResult(null);
      setAnalysisTrigger((prev) => prev + 1);

      const polyTimer = setTimeout(() => setShowPolygon(true), 1200);

      try {
        const data = await analyzeQuery({
          query: query.trim(),
          images: images && images.length > 0 ? images : undefined,
          useSamples: !images || images.length === 0 ? 'optical' : undefined,
        });

        setResult(data);

        if (!data.success) {
          const errText = data.error || 'Analysis failed';
          setError(errText);
          setMessages((prev) => [
            ...prev,
            { id: `a-${Date.now()}`, role: 'assistant', text: errText, error: true },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            {
              id: `a-${Date.now()}`,
              role: 'assistant',
              text: data.answer || 'No answer generated.',
              evidenceUrl: data.visual_evidence_b64 || undefined,
              changeMaskUrl: data.change_mask_b64 || undefined,
              confidence: data.confidence,
              models: data.models_used,
            },
          ]);
        }
      } catch (e) {
        const msg =
          e instanceof Error
            ? e.message
            : 'Failed to reach backend. Is API running on :8000?';
        setError(msg);
        setResult({ success: false, error: msg, confidence: 0 });
        setMessages((prev) => [
          ...prev,
          { id: `a-${Date.now()}`, role: 'assistant', text: msg, error: true },
        ]);
      } finally {
        clearTimeout(polyTimer);
        setIsAnalyzing(false);
        setShowPolygon(true);
      }
    },
    []
  );

  return (
    <div className="grid-bg flex h-screen w-screen overflow-hidden bg-white">
      <Sidebar />
      <CenterPanel
        onQuerySubmit={handleQuerySubmit}
        showPolygon={showPolygon}
        isAnalyzing={isAnalyzing}
        result={result}
        error={error}
        messages={messages}
      />
      <AgentPanel
        analysisTrigger={analysisTrigger}
        isAnalyzing={isAnalyzing}
        result={result}
      />
    </div>
  );
}