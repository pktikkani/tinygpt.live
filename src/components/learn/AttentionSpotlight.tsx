"use client";

import { useMemo, useState } from "react";
import { motion } from "motion/react";
import type { GPTModel } from "@/lib/gpt";
import type { LearnMode } from "./ChapterShell";
import ChapterShell from "./ChapterShell";
import Tiny from "./Tiny";

type Props = {
  model: GPTModel;
  mode: LearnMode;
  initialWord: string;
};

const TILE = 48;
const GAP = 10;
const ARC_H = 90;

export default function AttentionSpotlight({ model, mode, initialWord }: Props) {
  const [word, setWord] = useState(initialWord || "emma");
  const [head, setHead] = useState(0);
  const [queryPos, setQueryPos] = useState<number | null>(null);

  const analysis = useMemo(() => {
    if (!word) return null;
    return model.predictNext(word);
  }, [model, word]);

  // Display tokens: BOS + letters of the word
  const tokens = ["▶", ...word.split("")];
  const q = queryPos ?? tokens.length - 1;

  const headData = analysis?.attentionData.find((a) => a.layer === 0 && a.head === head);
  const weights = headData?.weights[q] ?? [];

  const top3 = analysis?.topTokens
    .filter((t) => t.char !== "<BOS>")
    .slice(0, 3) ?? [];

  const svgWidth = tokens.length * (TILE + GAP);
  const cx = (i: number) => i * (TILE + GAP) + TILE / 2;

  const handleWord = (raw: string) => {
    const clean = raw.toLowerCase().replace(/[^a-z]/g, "").slice(0, 10);
    setWord(clean);
    setQueryPos(null);
  };

  return (
    <ChapterShell
      emoji="🔦"
      title="The Attention Spotlight"
      mode={mode}
      simple={
        <>
          Here&apos;s Tiny&apos;s superpower: before guessing the next letter,
          it shines <b>spotlights back</b> on the letters it already read. Bright
          beams mean &quot;this letter matters a lot for my guess!&quot; This
          trick is called <b>attention</b> — it&apos;s the big idea inside
          ChatGPT. Click any letter to see where its spotlights point!
        </>
      }
      nerd={
        <>
          Real attention weights from the forward pass: each position emits a{" "}
          <b>query</b>, compares it against every earlier position&apos;s{" "}
          <b>key</b> (dot product, scaled by 1/√d), and softmaxes into weights
          that mix the <b>values</b>. 4 heads attend independently — switch
          heads and watch the pattern change. Arc opacity ∝ attention weight.
        </>
      }
    >
      <div className="rounded-lg border border-surface-border bg-surface p-6">
        <div className="mb-4 flex flex-wrap items-center justify-center gap-3">
          <input
            value={word}
            onChange={(e) => handleWord(e.target.value)}
            placeholder="type a word"
            className="w-44 rounded-lg border border-amber/40 bg-surface-light px-3 py-2 text-center tracking-widest outline-none focus:border-amber"
          />
          <div className="flex gap-1">
            {[0, 1, 2, 3].map((h) => (
              <button
                key={h}
                onClick={() => setHead(h)}
                className={`rounded px-3 py-1.5 text-xs font-bold transition-all ${
                  head === h
                    ? "border border-amber bg-amber/20 text-amber"
                    : "border border-surface-border bg-surface-light text-muted hover:text-foreground"
                }`}
              >
                {mode === "simple" ? `👁 ${h + 1}` : `H${h}`}
              </button>
            ))}
          </div>
        </div>

        {word.length > 0 ? (
          <div className="overflow-x-auto">
            <div className="mx-auto" style={{ width: svgWidth }}>
              {/* Attention arcs */}
              <svg width={svgWidth} height={ARC_H} className="block">
                {weights.map((w, k) => {
                  if (k >= q) {
                    // self-attention: small loop indicator
                    return k === q ? (
                      <motion.circle
                        key={`${word}-${head}-${q}-${k}`}
                        cx={cx(k)}
                        cy={ARC_H - 6}
                        r={5}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: Math.min(1, w * 1.2 + 0.05) }}
                      />
                    ) : null;
                  }
                  const x1 = cx(k);
                  const x2 = cx(q);
                  const mid = (x1 + x2) / 2;
                  const lift = Math.min(ARC_H - 10, 20 + (x2 - x1) * 0.25);
                  return (
                    <motion.path
                      key={`${word}-${head}-${q}-${k}`}
                      d={`M ${x1} ${ARC_H} Q ${mid} ${ARC_H - lift} ${x2} ${ARC_H}`}
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth={1.5 + w * 5}
                      strokeLinecap="round"
                      initial={{ pathLength: 0, opacity: 0 }}
                      animate={{ pathLength: 1, opacity: Math.min(1, w * 1.3 + 0.06) }}
                      transition={{ duration: 0.5, delay: k * 0.05 }}
                    />
                  );
                })}
              </svg>

              {/* Token tiles */}
              <div className="flex" style={{ gap: GAP }}>
                {tokens.map((t, i) => {
                  const w = i <= q ? (weights[i] ?? 0) : 0;
                  const isQuery = i === q;
                  return (
                    <motion.button
                      key={`${word}-${i}`}
                      onClick={() => setQueryPos(i === 0 ? null : i)}
                      whileHover={{ scale: 1.1 }}
                      animate={
                        isQuery
                          ? { scale: [1, 1.06, 1] }
                          : { scale: 1 }
                      }
                      transition={isQuery ? { duration: 1.4, repeat: Infinity } : {}}
                      style={{
                        width: TILE,
                        height: TILE,
                        backgroundColor: isQuery
                          ? "rgba(34,197,94,0.25)"
                          : `rgba(245,158,11,${Math.min(0.85, w)})`,
                      }}
                      className={`flex items-center justify-center rounded-lg border-2 text-lg font-bold ${
                        isQuery
                          ? "border-green text-green"
                          : i === 0
                            ? "border-surface-border text-muted text-xs"
                            : "border-surface-border"
                      }`}
                    >
                      {t}
                    </motion.button>
                  );
                })}
              </div>
              <p className="text-muted mt-2 text-center text-[10px]">
                {mode === "simple" ? (
                  <>
                    the <span className="text-green">green letter</span> is reading — brighter{" "}
                    <span className="text-amber">orange letters</span> are getting more spotlight
                    (▶ = the start marker)
                  </>
                ) : (
                  <>
                    query position {q} attending over keys 0..{q} · head {head} · layer 0 · click a tile to move the query
                  </>
                )}
              </p>
            </div>
          </div>
        ) : (
          <p className="text-muted py-8 text-center text-sm">type a word to see attention</p>
        )}

        {/* Prediction bars */}
        {word.length > 0 && top3.length > 0 && (
          <div className="mx-auto mt-6 max-w-sm">
            <p className="text-muted mb-2 text-center text-xs">
              {mode === "simple"
                ? `after looking, Tiny's top guesses for the letter after "${word}":`
                : "resulting next-token distribution (top 3):"}
            </p>
            {top3.map((t, i) => (
              <div key={t.id} className="mb-1.5 flex items-center gap-2">
                <span className="w-6 text-center font-bold text-amber">{t.char}</span>
                <div className="h-4 flex-1 overflow-hidden rounded bg-surface-light">
                  <motion.div
                    className="h-full rounded bg-green/60"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(2, t.prob * 100)}%` }}
                    transition={{ delay: i * 0.12, type: "spring", stiffness: 80 }}
                  />
                </div>
                <span className="text-muted w-10 text-right text-xs">
                  {(t.prob * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="mt-6 flex justify-center">
          <Tiny
            size={80}
            mood="thinking"
            say={
              mode === "simple"
                ? "Each of my 4 eyes looks for something different!"
                : "4 heads × 4 dims each = 16-dim attention output per position"
            }
          />
        </div>
      </div>
    </ChapterShell>
  );
}
