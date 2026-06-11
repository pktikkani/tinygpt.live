"use client";

import { useCallback, useMemo, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import Navigation from "@/components/Navigation";
import Tiny from "@/components/learn/Tiny";
import type { LearnMode } from "@/components/learn/ChapterShell";
import TokenizerPlay from "@/components/learn/TokenizerPlay";
import GuessGame from "@/components/learn/GuessGame";
import RobotSchool from "@/components/learn/RobotSchool";
import AttentionSpotlight from "@/components/learn/AttentionSpotlight";
import SillinessDial from "@/components/learn/SillinessDial";
import Finale from "@/components/learn/Finale";
import { GPTModel } from "@/lib/gpt";

const CHAPTERS = [
  { id: "welcome", label: "Meet Tiny" },
  { id: "tokens", label: "Numbers" },
  { id: "game", label: "The Game" },
  { id: "school", label: "School" },
  { id: "rematch", label: "Rematch" },
  { id: "attention", label: "Attention" },
  { id: "dial", label: "Silliness" },
  { id: "finale", label: "Finale" },
];

export default function LearnPage() {
  const model = useMemo(() => new GPTModel(42), []);
  const [chapter, setChapter] = useState(0);
  const [maxVisited, setMaxVisited] = useState(0);
  const [mode, setMode] = useState<LearnMode>("simple");
  const [playerName, setPlayerName] = useState("");

  const goTo = useCallback((c: number) => {
    const clamped = Math.max(0, Math.min(CHAPTERS.length - 1, c));
    setChapter(clamped);
    setMaxVisited((m) => Math.max(m, clamped));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const next = useCallback(() => goTo(chapter + 1), [goTo, chapter]);

  // Chapters that advance via their own "continue" button
  const selfAdvancing = ["game", "school", "rematch"].includes(CHAPTERS[chapter].id);
  const isFinale = CHAPTERS[chapter].id === "finale";

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="pt-14">
        <div className="mx-auto max-w-7xl px-6 py-8">
          {/* Header: progress + mode toggle */}
          <div className="mx-auto mb-8 flex max-w-3xl flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-1.5">
              {CHAPTERS.map((c, i) => (
                <button
                  key={c.id}
                  onClick={() => i <= maxVisited && goTo(i)}
                  disabled={i > maxVisited}
                  title={c.label}
                  className="group flex flex-col items-center"
                >
                  <motion.div
                    animate={{
                      scale: i === chapter ? 1.3 : 1,
                      backgroundColor:
                        i === chapter
                          ? "#f59e0b"
                          : i <= maxVisited
                            ? "#22c55e"
                            : "#33333355",
                    }}
                    className="h-2.5 w-2.5 rounded-full"
                  />
                  <span
                    className={`mt-1 hidden text-[8px] sm:block ${
                      i === chapter ? "text-amber font-bold" : "text-muted"
                    }`}
                  >
                    {c.label}
                  </span>
                </button>
              ))}
            </div>

            {/* Simple / Technical toggle */}
            <div className="flex rounded-lg border border-surface-border bg-surface p-0.5 text-xs">
              {(
                [
                  ["simple", "🧒 Simple"],
                  ["nerd", "🔬 Technical"],
                ] as const
              ).map(([m, label]) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`relative rounded-md px-3 py-1.5 transition-colors ${
                    mode === m ? "text-background" : "text-muted hover:text-foreground"
                  }`}
                >
                  {mode === m && (
                    <motion.div
                      layoutId="mode-pill"
                      className="absolute inset-0 rounded-md bg-amber"
                    />
                  )}
                  <span className="relative">{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Chapter content */}
          <AnimatePresence mode="wait">
            <motion.div key={chapter}>
              {CHAPTERS[chapter].id === "welcome" && <Welcome mode={mode} onStart={next} />}
              {CHAPTERS[chapter].id === "tokens" && (
                <TokenizerPlay model={model} mode={mode} onName={setPlayerName} />
              )}
              {CHAPTERS[chapter].id === "game" && (
                <GuessGame model={model} mode={mode} variant="untrained" onDone={next} />
              )}
              {CHAPTERS[chapter].id === "school" && (
                <RobotSchool
                  model={model}
                  mode={mode}
                  alreadyTrained={model.getCurrentStep() > 50}
                  onDone={next}
                />
              )}
              {CHAPTERS[chapter].id === "rematch" && (
                <GuessGame model={model} mode={mode} variant="trained" onDone={next} />
              )}
              {CHAPTERS[chapter].id === "attention" && (
                <AttentionSpotlight model={model} mode={mode} initialWord={playerName || "emma"} />
              )}
              {CHAPTERS[chapter].id === "dial" && <SillinessDial model={model} mode={mode} />}
              {CHAPTERS[chapter].id === "finale" && (
                <Finale mode={mode} playerName={playerName} />
              )}
            </motion.div>
          </AnimatePresence>

          {/* Footer nav */}
          <div className="mx-auto mt-8 flex max-w-3xl items-center justify-between pb-12">
            {chapter > 0 ? (
              <button
                onClick={() => goTo(chapter - 1)}
                className="text-muted text-sm transition-colors hover:text-foreground"
              >
                ← back
              </button>
            ) : (
              <span />
            )}
            {!isFinale &&
              chapter > 0 &&
              (selfAdvancing ? (
                <button
                  onClick={next}
                  className="text-muted text-xs transition-colors hover:text-foreground"
                >
                  skip →
                </button>
              ) : (
                <motion.button
                  onClick={next}
                  whileHover={{ scale: 1.05, x: 4 }}
                  className="rounded-lg border border-amber/40 bg-amber/10 px-5 py-2 text-sm font-bold text-amber transition-colors hover:bg-amber/20"
                >
                  next →
                </motion.button>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Welcome({ mode, onStart }: { mode: LearnMode; onStart: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto max-w-2xl text-center"
    >
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 200, damping: 14, delay: 0.2 }}
        className="mb-6 flex justify-center"
      >
        <Tiny
          mood="excited"
          size={150}
          say="Hi! I'm Tiny — a real AI living in your browser!"
        />
      </motion.div>

      <h1 className="text-amber glow-amber mb-3 text-3xl font-bold tracking-wide">
        How Does a Computer Learn to Write?
      </h1>
      <p className="text-muted mx-auto mb-2 max-w-lg text-sm leading-relaxed">
        {mode === "simple" ? (
          <>
            Tiny is a baby version of ChatGPT — same brain design, just{" "}
            <b>360 billion times smaller</b>. In the next few minutes you&apos;ll
            play games with Tiny, send it to school, and discover the real
            secrets behind how AI writes. No magic, promise!
          </>
        ) : (
          <>
            This is a complete GPT — multi-head attention, MLP blocks, residual
            connections, Adam optimizer — with ~5,000 parameters, ported from
            Karpathy&apos;s pure-Python GPT and running live in your browser.
            Every demo ahead uses real forward passes and real gradients, not
            canned animations.
          </>
        )}
      </p>
      <p className="text-muted/60 mb-8 text-xs">
        ages 6 to 106 · switch 🧒/🔬 anytime in the top corner
      </p>

      <motion.button
        onClick={onStart}
        whileHover={{ scale: 1.06 }}
        whileTap={{ scale: 0.94 }}
        animate={{
          boxShadow: [
            "0 0 0px rgba(245,158,11,0)",
            "0 0 30px rgba(245,158,11,0.4)",
            "0 0 0px rgba(245,158,11,0)",
          ],
        }}
        transition={{ duration: 2, repeat: Infinity }}
        className="rounded-2xl border-2 border-amber bg-amber/10 px-10 py-4 text-xl font-bold text-amber"
      >
        Start the adventure →
      </motion.button>
    </motion.div>
  );
}
