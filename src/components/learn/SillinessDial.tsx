"use client";

import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { GPTModel } from "@/lib/gpt";
import type { LearnMode } from "./ChapterShell";
import ChapterShell from "./ChapterShell";
import Tiny, { type TinyMood } from "./Tiny";

type Props = {
  model: GPTModel;
  mode: LearnMode;
};

function vibe(t: number): { emoji: string; label: string; mood: TinyMood } {
  if (t <= 0.25) return { emoji: "🥶", label: "super careful", mood: "sleepy" };
  if (t <= 0.55) return { emoji: "😌", label: "playing it safe", mood: "happy" };
  if (t <= 0.9) return { emoji: "🙂", label: "just right", mood: "happy" };
  if (t <= 1.2) return { emoji: "🤪", label: "getting silly", mood: "excited" };
  return { emoji: "🤯", label: "total chaos", mood: "excited" };
}

export default function SillinessDial({ model, mode }: Props) {
  const [temp, setTemp] = useState(0.7);
  const [names, setNames] = useState<{ text: string; key: number }[]>([]);
  const [genCount, setGenCount] = useState(0);
  const [busy, setBusy] = useState(false);

  const generate = useCallback(() => {
    setBusy(true);
    // let the button animation paint before the (synchronous) forward passes
    setTimeout(() => {
      const batch = model.generateBatch(5, temp);
      setNames(
        batch.map((r, i) => ({ text: r.text || "(speechless)", key: genCount * 5 + i }))
      );
      setGenCount((c) => c + 1);
      setBusy(false);
    }, 50);
  }, [model, temp, genCount]);

  const v = vibe(temp);

  return (
    <ChapterShell
      emoji="🎛️"
      title="The Silliness Dial"
      mode={mode}
      simple={
        <>
          One last secret: Tiny has a <b>silliness dial</b>! Turned down low,
          Tiny only picks letters it&apos;s really sure about — safe but boring
          names. Turned up high, Tiny takes wild chances — crazy, made-up
          names! Real chatbots have this exact dial. Try both extremes!
        </>
      }
      nerd={
        <>
          This is <b>temperature</b>: logits are divided by T before the
          softmax. T→0 sharpens the distribution toward argmax (deterministic,
          repetitive); T&gt;1 flattens it toward uniform (diverse, error-prone).
          The same parameter you set in every LLM API call.
        </>
      }
    >
      <div className="rounded-lg border border-surface-border bg-surface p-6">
        <div className="mb-6 flex justify-center">
          <Tiny
            mood={v.mood}
            say={`${v.emoji} Silliness: ${v.label}${mode === "nerd" ? ` (T=${temp.toFixed(2)})` : ""}`}
          />
        </div>

        {/* The dial */}
        <div className="mx-auto mb-2 max-w-md">
          <div className="mb-1 flex justify-between text-2xl">
            {["🥶", "😌", "🙂", "🤪", "🤯"].map((e, i) => (
              <motion.span
                key={e}
                animate={{
                  scale: vibe(0.1 + i * 0.35).label === v.label ? 1.5 : 1,
                  opacity: vibe(0.1 + i * 0.35).label === v.label ? 1 : 0.35,
                }}
              >
                {e}
              </motion.span>
            ))}
          </div>
          <input
            type="range"
            min={0.1}
            max={1.5}
            step={0.05}
            value={temp}
            onChange={(e) => setTemp(Number(e.target.value))}
            className="w-full accent-amber-500"
            style={{ accentColor: "#f59e0b" }}
          />
          <div className="text-muted flex justify-between text-[10px]">
            <span>careful</span>
            {mode === "nerd" && <span className="text-amber">temperature = {temp.toFixed(2)}</span>}
            <span>chaos</span>
          </div>
        </div>

        <div className="mb-6 text-center">
          <motion.button
            onClick={generate}
            disabled={busy}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.92 }}
            className="rounded-xl border-2 border-green bg-green/10 px-6 py-3 font-bold text-green disabled:opacity-50"
          >
            {busy ? "inventing..." : "🎲 Invent 5 names!"}
          </motion.button>
        </div>

        {/* Generated names */}
        <div className="mx-auto min-h-[140px] max-w-sm space-y-2">
          <AnimatePresence mode="popLayout">
            {names.map((n, i) => (
              <motion.div
                key={n.key}
                layout
                initial={{ opacity: 0, x: temp > 1 ? (i % 2 ? 60 : -60) : 0, y: temp <= 1 ? 16 : 0, scale: 0.7, rotate: temp > 1 ? (i % 2 ? 6 : -6) : 0 }}
                animate={{ opacity: 1, x: 0, y: 0, scale: 1, rotate: 0 }}
                exit={{ opacity: 0, scale: 0.6 }}
                transition={{ delay: i * 0.1, type: "spring", stiffness: 200, damping: 16 }}
                className="rounded-lg border border-surface-border bg-surface-light px-4 py-2 text-center text-lg font-bold tracking-widest"
              >
                {n.text}
              </motion.div>
            ))}
          </AnimatePresence>
          {names.length === 0 && (
            <p className="text-muted pt-10 text-center text-sm">
              set the dial, then hit the button!
            </p>
          )}
        </div>

        {names.length > 0 && (
          <p className="text-muted mt-4 text-center text-xs">
            {mode === "simple"
              ? "Try the dial at both ends and invent again — see how the names change?"
              : "Low T → the model keeps re-sampling its modal letters. High T → low-probability tokens sneak in."}
          </p>
        )}
      </div>
    </ChapterShell>
  );
}
