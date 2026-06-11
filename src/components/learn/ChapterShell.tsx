"use client";

import { motion } from "motion/react";
import type { ReactNode } from "react";

export type LearnMode = "simple" | "nerd";

type Props = {
  emoji: string;
  title: string;
  /** Friendly explanation, for everyone */
  simple: ReactNode;
  /** The technical version, for the curious */
  nerd: ReactNode;
  mode: LearnMode;
  children: ReactNode;
};

export default function ChapterShell({ emoji, title, simple, nerd, mode, children }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -24 }}
      transition={{ duration: 0.4 }}
      className="mx-auto w-full max-w-3xl"
    >
      <div className="mb-4 flex items-center gap-3">
        <motion.span
          className="text-4xl"
          initial={{ scale: 0, rotate: -30 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 260, damping: 14, delay: 0.15 }}
        >
          {emoji}
        </motion.span>
        <h2 className="text-amber glow-amber text-xl font-bold tracking-wide sm:text-2xl">{title}</h2>
      </div>

      <motion.div
        key={mode}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={`mb-6 rounded-lg border px-4 py-3 text-sm leading-relaxed ${
          mode === "simple"
            ? "border-amber/30 bg-amber/5"
            : "border-green/30 bg-green/5"
        }`}
      >
        {mode === "simple" ? simple : nerd}
      </motion.div>

      {children}
    </motion.div>
  );
}
