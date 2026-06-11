"use client";

import { useMemo } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import type { LearnMode } from "./ChapterShell";
import Tiny from "./Tiny";

type Props = {
  mode: LearnMode;
  playerName: string;
};

const CONFETTI_COLORS = ["#f59e0b", "#22c55e", "#60a5fa", "#f87171", "#a78bfa"];

// Deterministic pseudo-random so renders are stable
function rand(i: number, salt: number): number {
  const x = Math.sin(i * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

const RECAP = [
  { emoji: "🔢", simple: "Letters become secret numbers", nerd: "Tokenization → integer IDs" },
  { emoji: "🎮", simple: "The whole game is guessing the next letter", nerd: "Next-token prediction, P(xₜ₊₁|x₁..xₜ)" },
  { emoji: "🎒", simple: "Practice on examples makes the guesses smart", nerd: "Cross-entropy loss + backprop + Adam" },
  { emoji: "🔦", simple: "Spotlights on earlier letters = attention", nerd: "Scaled dot-product multi-head attention" },
  { emoji: "🎛️", simple: "The silliness dial controls how wild it gets", nerd: "Temperature scaling of logits" },
];

export default function Finale({ mode, playerName }: Props) {
  const confetti = useMemo(
    () =>
      Array.from({ length: 50 }, (_, i) => ({
        left: rand(i, 1) * 100,
        delay: rand(i, 2) * 4,
        duration: 3 + rand(i, 3) * 3,
        color: CONFETTI_COLORS[i % CONFETTI_COLORS.length],
        size: 6 + rand(i, 4) * 8,
        rotate: rand(i, 5) * 360,
      })),
    []
  );

  const displayName = playerName
    ? playerName[0].toUpperCase() + playerName.slice(1)
    : "Friend";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="relative mx-auto w-full max-w-3xl overflow-hidden"
    >
      {/* Confetti */}
      <div className="pointer-events-none absolute inset-0 z-10">
        {confetti.map((c, i) => (
          <motion.div
            key={i}
            className="absolute rounded-sm"
            style={{
              left: `${c.left}%`,
              width: c.size,
              height: c.size * 0.6,
              backgroundColor: c.color,
            }}
            initial={{ top: -20, rotate: 0, opacity: 1 }}
            animate={{ top: "110%", rotate: c.rotate + 720, opacity: [1, 1, 0.6] }}
            transition={{ duration: c.duration, delay: c.delay, repeat: Infinity, ease: "linear" }}
          />
        ))}
      </div>

      <div className="relative text-center">
        <motion.h2
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 12 }}
          className="text-amber glow-amber mb-2 text-3xl font-bold"
        >
          🎉 You did it, {displayName}!
        </motion.h2>
        <p className="text-muted mb-6 text-sm">
          {mode === "simple"
            ? "You just learned how ChatGPT really works. Not a cartoon version — the real thing, running in your browser!"
            : "Everything you just played with — tokenizer, attention, backprop, temperature — is the same algorithm behind GPT-4, just ~360 billion times smaller."}
        </p>

        <div className="mb-6 flex justify-center">
          <Tiny mood="excited" size={130} say="You know my secrets now! 🤖❤️" />
        </div>

        {/* Recap cards */}
        <div className="mb-8 space-y-2 text-left">
          {RECAP.map((r, i) => (
            <motion.div
              key={r.emoji}
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + i * 0.18 }}
              className="flex items-center gap-3 rounded-lg border border-surface-border bg-surface px-4 py-3"
            >
              <span className="text-2xl">{r.emoji}</span>
              <span className="text-sm">{mode === "simple" ? r.simple : r.nerd}</span>
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.6 + i * 0.18, type: "spring" }}
                className="text-green ml-auto font-bold"
              >
                ✓
              </motion.span>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.6 }}
          className="flex flex-wrap justify-center gap-3 pb-8"
        >
          <Link
            href="/"
            className="rounded-xl border-2 border-amber bg-amber/10 px-6 py-3 font-bold text-amber transition-colors hover:bg-amber/20"
          >
            🛠 Explore the full playground
          </Link>
          <a
            href="https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-xl border border-surface-border bg-surface px-6 py-3 text-sm text-muted transition-colors hover:border-green/50 hover:text-green"
          >
            Read Karpathy&apos;s original code →
          </a>
        </motion.div>
      </div>
    </motion.div>
  );
}
