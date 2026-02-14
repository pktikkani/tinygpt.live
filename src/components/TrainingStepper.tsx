"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { GPTModel, TrainStepResult } from "@/lib/gpt";

type Props = {
  model: GPTModel;
  onStep: (result: TrainStepResult) => void;
};

type LossPoint = { step: number; loss: number };

export default function TrainingStepper({ model, onStep }: Props) {
  const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
  const [lastResult, setLastResult] = useState<TrainStepResult | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [autoTrain, setAutoTrain] = useState(false);
  const autoTrainRef = useRef(false);
  const [batchSize, setBatchSize] = useState(1);
  const [targetLoss, setTargetLoss] = useState<number | null>(null);
  const targetLossRef = useRef<number | null>(null);
  const [stoppedReason, setStoppedReason] = useState("");
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    setIsDark(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsDark(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const stopAutoTrain = useCallback(() => {
    autoTrainRef.current = false;
    setAutoTrain(false);
  }, []);

  const doStep = useCallback(() => {
    const result = model.trainStep(5000);
    setLastResult(result);
    setLossHistory((prev) => [
      ...prev,
      { step: result.step, loss: result.loss },
    ]);
    onStep(result);
    return result;
  }, [model, onStep]);

  const handleStep = useCallback(() => {
    setIsTraining(true);
    setStoppedReason("");
    setTimeout(() => {
      for (let i = 0; i < batchSize; i++) {
        doStep();
      }
      setIsTraining(false);
    }, 10);
  }, [doStep, batchSize]);

  const handleAutoTrain = useCallback(() => {
    if (autoTrainRef.current) {
      stopAutoTrain();
      return;
    }

    autoTrainRef.current = true;
    setAutoTrain(true);
    setStoppedReason("");

    const loop = () => {
      if (!autoTrainRef.current) return;
      const result = doStep();

      // Check if we hit the target loss
      const tl = targetLossRef.current;
      if (tl !== null && result.loss <= tl) {
        stopAutoTrain();
        setStoppedReason(`reached target loss ${tl}`);
        return;
      }

      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }, [doStep, stopAutoTrain]);

  const currentStep = lastResult?.step ?? 0;
  const currentLoss = lastResult?.loss;
  const lossDisplay =
    currentLoss == null
      ? "—"
      : isNaN(currentLoss)
        ? "NaN"
        : currentLoss.toFixed(4);
  const lossColor =
    currentLoss != null && !isNaN(currentLoss) && currentLoss < 2
      ? "text-green"
      : "text-amber";

  return (
    <div className="rounded-lg border border-surface-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-amber text-sm font-bold tracking-wider uppercase">
          Training
        </h3>
        <div className="flex items-center gap-4 text-xs">
          <span className="text-muted">
            step{" "}
            <span className="text-green font-bold">{currentStep}</span>
          </span>
          <span className="text-muted">
            loss{" "}
            <span className={`font-bold ${lossColor}`}>{lossDisplay}</span>
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={handleStep}
          disabled={isTraining || autoTrain}
          className="flex-1 rounded border border-green/30 bg-green/10 px-3 py-2 text-sm text-green font-medium transition-all hover:bg-green/20 hover:border-green/50 disabled:opacity-50"
        >
          {isTraining
            ? "training..."
            : batchSize === 1
              ? "Step"
              : `Step x${batchSize}`}
        </button>

        <button
          onClick={handleAutoTrain}
          className={`rounded px-3 py-2 text-sm font-medium transition-all ${
            autoTrain
              ? "border border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20"
              : "border border-amber/30 bg-amber/10 text-amber hover:bg-amber/20"
          }`}
        >
          {autoTrain ? "Stop" : "Auto"}
        </button>

        <select
          value={batchSize}
          onChange={(e) => setBatchSize(Number(e.target.value))}
          className="rounded border border-surface-border bg-surface-light px-2 py-1 text-xs text-muted"
        >
          <option value={1}>x1</option>
          <option value={5}>x5</option>
          <option value={10}>x10</option>
          <option value={25}>x25</option>
        </select>
      </div>

      {/* Stop at target loss */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-muted text-[10px] shrink-0">stop at loss</span>
        <select
          value={targetLoss ?? "none"}
          onChange={(e) => {
            const val = e.target.value === "none" ? null : Number(e.target.value);
            setTargetLoss(val);
            targetLossRef.current = val;
          }}
          className="rounded border border-surface-border bg-surface-light px-2 py-1 text-xs text-muted"
        >
          <option value="none">never</option>
          <option value={2.0}>2.0</option>
          <option value={1.5}>1.5</option>
          <option value={1.2}>1.2</option>
          <option value={1.0}>1.0</option>
          <option value={0.9}>0.9</option>
          <option value={0.8}>0.8</option>
          <option value={0.5}>0.5</option>
        </select>
        {targetLoss !== null && (
          <span className="text-amber text-[10px]">
            auto-train stops when loss hits {targetLoss}
          </span>
        )}
      </div>

      {/* Stopped reason */}
      {stoppedReason && (
        <div className="mb-3 rounded bg-green/10 border border-green/20 px-3 py-2 text-xs text-green">
          {stoppedReason}
        </div>
      )}

      {/* Training info */}
      {lastResult && (
        <div className="mb-3 rounded bg-surface-light px-3 py-2 text-xs">
          <span className="text-muted">training on: </span>
          <span className="text-green font-medium">
            &quot;{lastResult.doc}&quot;
          </span>
        </div>
      )}

      {/* Loss chart */}
      {lossHistory.length > 1 && (
        <div className="h-44">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lossHistory}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={isDark ? "#1e1e1e" : "#e5e7eb"}
                vertical={false}
              />
              <XAxis
                dataKey="step"
                stroke={isDark ? "#666" : "#9ca3af"}
                tick={{ fontSize: 10 }}
                tickLine={false}
              />
              <YAxis
                stroke={isDark ? "#666" : "#9ca3af"}
                tick={{ fontSize: 10 }}
                tickLine={false}
                domain={["auto", "auto"]}
              />
              <Tooltip
                contentStyle={{
                  background: isDark ? "#141414" : "#ffffff",
                  border: `1px solid ${isDark ? "#1e1e1e" : "#e5e7eb"}`,
                  borderRadius: 6,
                  fontSize: 11,
                  color: isDark ? "#e0e0e0" : "#1a1a1a",
                }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#22c55e"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {lossHistory.length === 0 && (
        <div className="flex h-44 items-center justify-center text-muted text-sm">
          Click &quot;Step&quot; to begin training
        </div>
      )}
    </div>
  );
}
