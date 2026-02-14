"use client";

import { useState, useCallback } from "react";
import { motion } from "motion/react";
import type { GPTModel, GenerateResult } from "@/lib/gpt";

type Props = {
  model: GPTModel;
  onGenerate: (results: GenerateResult[]) => void;
};

export default function TemperatureControl({ model, onGenerate }: Props) {
  const [temperature, setTemperature] = useState(0.5);
  const [samples, setSamples] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = useCallback(() => {
    setIsGenerating(true);
    // Use setTimeout to let UI update
    setTimeout(() => {
      const results = model.generateBatch(8, temperature);
      setSamples(results.map((r) => r.text));
      onGenerate(results);
      setIsGenerating(false);
    }, 10);
  }, [model, temperature, onGenerate]);

  const tempLabel =
    temperature < 0.3
      ? "frigid"
      : temperature < 0.5
        ? "cool"
        : temperature < 0.7
          ? "warm"
          : temperature < 0.9
            ? "hot"
            : "molten";

  return (
    <div className="rounded-lg border border-surface-border bg-surface p-4">
      <h3 className="text-amber text-sm font-bold tracking-wider uppercase mb-3">
        Temperature Control
      </h3>

      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-muted text-xs">conservative</span>
          <span
            className="text-sm font-bold"
            style={{
              color: `hsl(${(1 - temperature) * 120}, 80%, 50%)`,
            }}
          >
            {temperature.toFixed(2)}{" "}
            <span className="text-muted text-[10px]">({tempLabel})</span>
          </span>
          <span className="text-muted text-xs">creative</span>
        </div>

        <input
          type="range"
          min={0.05}
          max={1.5}
          step={0.05}
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          className="w-full accent-amber"
          style={{
            background: `linear-gradient(to right, var(--color-green), var(--color-amber) ${
              (temperature / 1.5) * 100
            }%, var(--color-surface-border) ${(temperature / 1.5) * 100}%)`,
          }}
        />

        <div className="text-muted mt-1 text-[10px]">
          Low = predictable, common names. High = wild, creative names.
        </div>
      </div>

      <button
        onClick={handleGenerate}
        disabled={isGenerating}
        className="w-full rounded border border-amber/30 bg-amber/10 px-4 py-2 text-sm text-amber font-medium transition-all hover:bg-amber/20 hover:border-amber/50 disabled:opacity-50"
      >
        {isGenerating ? "generating..." : "Generate Names"}
      </button>

      {samples.length > 0 && (
        <div className="mt-3 space-y-1">
          {samples.map((name, i) => (
            <motion.div
              key={`${name}-${i}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="flex items-center gap-2 rounded px-2 py-1 bg-surface-light"
            >
              <span className="text-green text-[10px]">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="text-foreground text-sm font-medium tracking-wide">
                {name || "(empty)"}
              </span>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
