"use client";

import { useState } from "react";
import { motion } from "motion/react";
import type { AttentionData } from "@/lib/gpt";

type Props = {
  attentionData: AttentionData[];
  tokenChars: string[];
  nHead: number;
};

export default function AttentionHeatmap({
  attentionData,
  tokenChars,
  nHead,
}: Props) {
  const [selectedHead, setSelectedHead] = useState(0);

  if (!attentionData.length || !tokenChars.length) {
    return (
      <div className="rounded-lg border border-surface-border bg-surface p-4">
        <h3 className="text-amber text-sm font-bold tracking-wider uppercase mb-3">
          Attention Heads
        </h3>
        <div className="text-muted flex h-40 items-center justify-center text-sm">
          Train or generate to see attention patterns
        </div>
      </div>
    );
  }

  // Find attention data for selected head
  const headData = attentionData.find(
    (a) => a.layer === 0 && a.head === selectedHead
  );
  const weights = headData?.weights ?? [];

  // Tokens to display (limit to what we have attention for)
  const displayTokens = tokenChars.slice(0, weights.length + 1);

  return (
    <div className="rounded-lg border border-surface-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-amber text-sm font-bold tracking-wider uppercase">
          Attention Heads
        </h3>
        <span className="text-muted text-xs">layer 0</span>
      </div>

      {/* Head selector tabs */}
      <div className="mb-3 flex gap-1">
        {Array.from({ length: nHead }, (_, h) => (
          <button
            key={h}
            onClick={() => setSelectedHead(h)}
            className={`rounded px-3 py-1 text-xs font-medium transition-all ${
              selectedHead === h
                ? "bg-green text-black"
                : "border border-surface-border text-muted hover:text-foreground hover:border-green/50"
            }`}
          >
            H{h}
          </button>
        ))}
      </div>

      {/* Heatmap grid */}
      <div className="overflow-x-auto">
        <div className="inline-block">
          {/* Column headers (key tokens) */}
          <div className="flex">
            <div className="w-10 shrink-0" />
            {displayTokens.map((ch, i) => (
              <div
                key={`col-${i}`}
                className="text-muted flex w-8 items-center justify-center text-[10px]"
              >
                {ch === "<BOS>" ? "^" : ch}
              </div>
            ))}
          </div>

          {/* Rows (query tokens) */}
          {weights.map((row, qi) => (
            <div key={qi} className="flex items-center">
              {/* Row label */}
              <div className="text-muted w-10 shrink-0 text-right text-[10px] pr-2">
                {displayTokens[qi + 1] === "<BOS>"
                  ? "^"
                  : displayTokens[qi + 1] ?? "?"}
              </div>

              {/* Attention cells */}
              {row.map((weight, ki) => {
                const intensity = Math.min(weight, 1);
                return (
                  <motion.div
                    key={`${qi}-${ki}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: (qi * row.length + ki) * 0.01 }}
                    className="flex h-8 w-8 items-center justify-center text-[9px] font-medium rounded-sm m-px cursor-default"
                    style={{
                      background: `rgba(34, 197, 94, ${intensity * 0.8})`,
                      color:
                        intensity > 0.5
                          ? "rgba(0,0,0,0.8)"
                          : "var(--color-muted)",
                    }}
                    title={`Q:${displayTokens[qi + 1]} → K:${displayTokens[ki]} = ${weight.toFixed(3)}`}
                  >
                    {weight > 0.01 ? weight.toFixed(1) : ""}
                  </motion.div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="text-muted mt-2 text-[10px]">
        Rows = query token, Cols = key token. Brighter = higher attention.
      </div>
    </div>
  );
}
