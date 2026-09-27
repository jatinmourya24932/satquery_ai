'use client';

import { useState } from 'react';
import {
  LayoutDashboard,
  Satellite,
  Layers,
  Sparkles,
  Clock,
  Settings,
  Radio,
  Orbit,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, active: true },
  { id: 'satellite', label: 'Satellite Explorer', icon: Satellite },
  { id: 'layers', label: 'Analytics Layers', icon: Layers },
  { id: 'ai', label: 'Query with AI', icon: Sparkles },
  { id: 'timeseries', label: 'Time-Series', icon: Clock },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const [activeId, setActiveId] = useState('overview');

  return (
    <aside className="flex h-full w-60 flex-col border-r border-slate-200 bg-slate-100">
      {/* Logo */}
      <div className="flex items-center gap-3 border-b border-slate-200 bg-white px-5 py-5">
        <div className="relative flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600/10 to-emerald-700/10 ring-1 ring-blue-600/20">
          <Orbit className="h-5 w-5 text-blue-700" />
          <div className="absolute inset-0 rounded-lg ring-1 ring-blue-600/10 animate-glow" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-bold text-slate-900 tracking-wide">
            SatQuery<span className="text-blue-700"> AI</span>
          </div>
          <div className="text-[10px] text-slate-500 tracking-wider uppercase">
            Geo Intel Dashboard
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-1 p-3">
        <div className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
          Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeId === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveId(item.id)}
              className={cn(
                'group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all duration-200',
                isActive
                  ? 'bg-blue-600 text-white shadow-[0_2px_12px_rgba(30,64,175,0.2)]'
                  : 'text-slate-600 hover:bg-slate-200/70 hover:text-slate-900'
              )}
            >
              {isActive && (
                <div className="absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-r-full bg-white" />
              )}
              <Icon
                className={cn(
                  'h-4 w-4 transition-transform group-hover:scale-110',
                  isActive ? 'text-white' : 'text-slate-500'
                )}
              />
              <span className="font-medium">{item.label}</span>
              {isActive && (
                <div className="ml-auto h-1.5 w-1.5 rounded-full bg-white animate-pulse-live" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Mission Context */}
      <div className="border-t border-slate-200 bg-white p-4">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5">
          <div className="mb-2 flex items-center gap-2">
            <Radio className="h-3.5 w-3.5 text-emerald-700" />
            <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Mission Context
            </span>
          </div>
          <div className="text-sm font-bold text-slate-900">
            Resourcesat-2A
          </div>
          <div className="text-[11px] text-slate-500">
            LISS-III Sensor · 23.5m Resolution
          </div>
          <div className="mt-2.5 flex items-center gap-2">
            <div className="flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2.5 py-1 ring-1 ring-emerald-600/30">
              <div className="h-2 w-2 rounded-full bg-emerald-600 animate-pulse-live" />
              <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700">
                Active
              </span>
            </div>
            <span className="text-[10px] text-slate-400">Orbit: 817km</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
