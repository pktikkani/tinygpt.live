"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { GPTModel } from "@/lib/gpt";

type Props = {
  model: GPTModel;
};

type TokenState = {
  char: string;
  id: number;
  stage: "input" | "embedding" | "attention" | "mlp" | "output";
};

const stages: { key: TokenState["stage"]; label: string; color: string }[] = [
  { key: "input", label: "Input", color: "#f59e0b" },
  { key: "embedding", label: "Embed", color: "#f59e0b" },
  { key: "attention", label: "Attn", color: "#22c55e" },
  { key: "mlp", label: "MLP", color: "#3b82f6" },
  { key: "output", label: "Output", color: "#a855f7" },
];

export default function TokenFlow({ model }: Props) {
  const [inputText, setInputText] = useState("");
  const [tokens, setTokens] = useState<TokenState[]>([]);
  const [outputTokens, setOutputTokens] = useState<string[]>([]);
  const [isFlowing, setIsFlowing] = useState(false);
  const [currentStageIdx, setCurrentStageIdx] = useState(-1);
  const [statusText, setStatusText] = useState("");

  const handleFlow = useCallback(() => {
    const text = inputText.trim() || "emma";
    setIsFlowing(true);
    setOutputTokens([]);
    setStatusText("tokenizing...");

    const chars = text.split("");
    const initialTokens: TokenState[] = chars.map((ch) => ({
      char: ch,
      id: model.tokenizer.uchars.indexOf(ch),
      stage: "input" as const,
    }));
    setTokens(initialTokens);
    setCurrentStageIdx(0);

    let idx = 1;
    const advanceStage = () => {
      if (idx >= stages.length) {
        // Generate continuation
        setStatusText("generating...");
        setTimeout(() => {
          try {
            const result = model.generate(0.5, 8);
            setOutputTokens(result.tokenChars);
            setStatusText(
              result.tokenChars.length > 0
                ? `generated "${result.text}"`
                : "model produced empty output — try training first"
            );
          } catch {
            setStatusText("generation failed — try training first");
          }
          setIsFlowing(false);
        }, 100);
        return;
      }

      const stage = stages[idx];
      setCurrentStageIdx(idx);
      setStatusText(`${stage.label.toLowerCase()}...`);
      setTokens((prev) => prev.map((t) => ({ ...t, stage: stage.key })));
      idx++;
      setTimeout(advanceStage, 500);
    };

    setTimeout(advanceStage, 400);
  }, [inputText, model]);

  const currentStage = stages[currentStageIdx] ?? stages[0];
  const progress =
    currentStageIdx >= 0
      ? ((currentStageIdx + 1) / stages.length) * 100
      : 0;

  return (
    <div className="rounded-lg border border-surface-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-amber text-sm font-bold tracking-wider uppercase">
          Token Flow
        </h3>
        {statusText && (
          <span className="text-muted text-[10px]">{statusText}</span>
        )}
      </div>

      {/* Input */}
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value.toLowerCase())}
          placeholder="type a name..."
          maxLength={12}
          className="flex-1 rounded border border-surface-border bg-surface-light px-3 py-2 text-sm text-foreground placeholder:text-muted focus:outline-none focus:border-amber/50"
        />
        <button
          onClick={handleFlow}
          disabled={isFlowing}
          className="rounded border border-amber/30 bg-amber/10 px-4 py-2 text-sm text-amber font-medium transition-all hover:bg-amber/20 disabled:opacity-50"
        >
          {isFlowing ? "flowing..." : "Flow"}
        </button>
      </div>

      {/* Pipeline */}
      <div className="relative">
        {/* Stage indicators */}
        <div className="flex justify-between mb-2">
          {stages.map((stage, i) => (
            <div
              key={stage.key}
              className="flex flex-col items-center gap-1"
            >
              <div
                className="h-3 w-3 rounded-full border-2 transition-all duration-300"
                style={{
                  borderColor: stage.color,
                  background:
                    i <= currentStageIdx ? stage.color : "transparent",
                  opacity: i <= currentStageIdx ? 1 : 0.3,
                }}
              />
              <span
                className="text-[10px] font-medium transition-opacity duration-300"
                style={{
                  color: stage.color,
                  opacity: i <= currentStageIdx ? 1 : 0.3,
                }}
              >
                {stage.label}
              </span>
            </div>
          ))}
        </div>

        {/* Progress bar */}
        <div className="relative h-1.5 rounded-full bg-surface-light mb-4 overflow-hidden">
          <motion.div
            className="absolute inset-y-0 left-0 rounded-full"
            style={{ background: currentStage.color }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>

        {/* Token chips */}
        <div className="flex flex-wrap gap-2 min-h-[44px] items-center">
          <AnimatePresence mode="popLayout">
            {tokens.map((token, i) => {
              const stage = stages.find((s) => s.key === token.stage)!;
              return (
                <motion.div
                  key={`${token.char}-${i}`}
                  layout
                  initial={{ opacity: 0, scale: 0.5, y: 10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.5 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center gap-1.5 rounded border px-2.5 py-1.5"
                  style={{
                    borderColor: `${stage.color}40`,
                    background: `${stage.color}10`,
                  }}
                >
                  <span
                    className="text-sm font-bold"
                    style={{ color: stage.color }}
                  >
                    {token.char}
                  </span>
                  <span className="text-[9px] text-muted">
                    {token.id >= 0 ? token.id : "?"}
                  </span>
                </motion.div>
              );
            })}
          </AnimatePresence>

          {/* Output tokens */}
          <AnimatePresence>
            {outputTokens.length > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-1.5"
              >
                <span className="text-muted text-sm mx-1">&rarr;</span>
                {outputTokens.map((ch, i) => (
                  <motion.span
                    key={`out-${i}`}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className="rounded border border-purple-500/30 bg-purple-500/10 px-2.5 py-1.5 text-sm font-bold text-purple-400"
                  >
                    {ch}
                  </motion.span>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <p className="text-muted mt-3 text-[10px]">
        Type a name and click Flow to watch tokens pass through each layer of the GPT.
        After the pipeline completes, the model generates a continuation.
      </p>
    </div>
  );
}
