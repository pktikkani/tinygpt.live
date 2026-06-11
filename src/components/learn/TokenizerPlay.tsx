"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { GPTModel } from "@/lib/gpt";
import type { LearnMode } from "./ChapterShell";
import ChapterShell from "./ChapterShell";
import Tiny from "./Tiny";

type Props = {
  model: GPTModel;
  mode: LearnMode;
  onName: (name: string) => void;
};

export default function TokenizerPlay({ model, mode, onName }: Props) {
  const [text, setText] = useState("");

  const handleChange = (raw: string) => {
    const clean = raw.toLowerCase().replace(/[^a-z]/g, "").slice(0, 12);
    setText(clean);
    if (clean.length >= 2) onName(clean);
  };

  const ids = text ? model.tokenizer.encode(text).slice(1, -1) : [];

  return (
    <ChapterShell
      emoji="🔢"
      title="Letters Become Numbers"
      mode={mode}
      simple={
        <>
          Tiny can&apos;t read letters — robots only understand <b>numbers</b>!
          So the very first thing Tiny does is swap every letter for its own
          secret number. Type your name below and watch the magic happen. ✨
        </>
      }
      nerd={
        <>
          This is <b>tokenization</b>. Our vocabulary is just 27 tokens: the
          letters a–z plus a special <code>&lt;BOS&gt;</code> (beginning of
          sequence) marker. Big models like GPT-4 use ~100k tokens covering
          word chunks (BPE), but the idea is identical: text → integer IDs
          that index into an embedding table.
        </>
      }
    >
      <div className="rounded-lg border border-surface-border bg-surface p-6">
        <div className="mb-6 flex justify-center">
          <Tiny
            mood={text.length > 0 ? "excited" : "happy"}
            say={
              text.length === 0
                ? "Type your name! I'll show you my secret code."
                : text.length < 3
                  ? "Keep going..."
                  : `Ooh, "${text}" looks delicious as numbers!`
            }
          />
        </div>

        <input
          value={text}
          onChange={(e) => handleChange(e.target.value)}
          placeholder="type your name (a–z)"
          autoFocus
          className="mx-auto block w-full max-w-sm rounded-lg border border-amber/40 bg-surface-light px-4 py-3 text-center text-xl tracking-[0.3em] outline-none focus:border-amber"
        />

        {/* Letter → number animation */}
        <div className="mt-8 flex min-h-[120px] flex-wrap items-center justify-center gap-2">
          <AnimatePresence mode="popLayout">
            {text.split("").map((ch, i) => (
              <motion.div
                key={`${i}-${ch}`}
                layout
                initial={{ opacity: 0, y: -30, scale: 0.5 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="flex flex-col items-center gap-1"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-amber/40 bg-amber/10 text-xl font-bold text-amber">
                  {ch}
                </div>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1, rotate: [0, 180, 360] }}
                  transition={{ delay: 0.2 + i * 0.06, duration: 0.4 }}
                  className="text-muted text-xs"
                >
                  ↓
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, scale: 0, rotateX: 90 }}
                  animate={{ opacity: 1, scale: 1, rotateX: 0 }}
                  transition={{ delay: 0.35 + i * 0.06, type: "spring", stiffness: 260 }}
                  className="flex h-12 w-12 items-center justify-center rounded-lg border border-green/40 bg-green/10 text-lg font-bold text-green"
                >
                  {ids[i]}
                </motion.div>
              </motion.div>
            ))}
          </AnimatePresence>

          {text.length === 0 && (
            <p className="text-muted text-sm">your letters will appear here...</p>
          )}
        </div>

        {text.length >= 2 && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-4 text-center text-xs text-muted"
          >
            {mode === "simple" ? (
              <>
                That&apos;s it! To Tiny, &quot;{text}&quot; is just{" "}
                <span className="text-green font-bold">[{ids.join(", ")}]</span>
              </>
            ) : (
              <>
                <code className="text-green">
                  encode(&quot;{text}&quot;) → [{model.tokenizer.BOS}, {ids.join(", ")},{" "}
                  {model.tokenizer.BOS}]
                </code>{" "}
                — wrapped in BOS tokens so the model knows where names start and end
              </>
            )}
          </motion.p>
        )}
      </div>
    </ChapterShell>
  );
}
