"use client";

import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { GPTModel } from "@/lib/gpt";
import type { LearnMode } from "./ChapterShell";
import ChapterShell from "./ChapterShell";
import Tiny, { type TinyMood } from "./Tiny";

type Props = {
  model: GPTModel;
  mode: LearnMode;
  alreadyTrained: boolean;
  onDone: () => void;
};

const MAX_STEPS = 300;
const TARGET_LOSS = 1.15;
const START_LOSS = 3.3; // ~ln(27), random guessing

export default function RobotSchool({ model, mode, alreadyTrained, onDone }: Props) {
  const [phase, setPhase] = useState<"idle" | "training" | "done">(
    alreadyTrained ? "done" : "idle"
  );
  const [beforeNames, setBeforeNames] = useState<string[]>([]);
  const [afterNames, setAfterNames] = useState<string[]>([]);
  const [loss, setLoss] = useState(START_LOSS);
  const [step, setStep] = useState(0);
  const [studying, setStudying] = useState("");
  const runningRef = useRef(false);

  const startSchool = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true;

    // Snapshot what the untrained brain produces
    const before = model.generateBatch(5, 0.8).map((r) => r.text || "(silence)");
    setBeforeNames(before);
    setPhase("training");

    let smoothLoss = START_LOSS;
    const loop = () => {
      const result = model.trainStep(MAX_STEPS + 50);
      smoothLoss = smoothLoss * 0.85 + result.loss * 0.15;
      setLoss(smoothLoss);
      setStep(result.step);
      setStudying(result.doc);

      if (result.step >= MAX_STEPS || smoothLoss <= TARGET_LOSS) {
        runningRef.current = false;
        setAfterNames(model.generateBatch(5, 0.8).map((r) => r.text || "(silence)"));
        setPhase("done");
        return;
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }, [model]);

  // Confusion: 100% at random-guessing loss, 0% at fully-learned
  const confusion = Math.max(0, Math.min(1, (loss - 0.9) / (START_LOSS - 0.9)));
  const confusionPct = Math.round(confusion * 100);

  const tinyMood: TinyMood =
    phase === "idle" ? "sleepy" : phase === "training" ? "studying" : "excited";

  return (
    <ChapterShell
      emoji="🎒"
      title="Robot School"
      mode={mode}
      simple={
        <>
          How does Tiny get smart? <b>Practice!</b> Tiny reads real names one
          by one, tries to guess every next letter, and every time it&apos;s
          wrong, its brain adjusts a tiny bit. Do that hundreds of times and
          the guesses stop being random. Watch the{" "}
          <b>confusion meter</b> go down as Tiny studies!
        </>
      }
      nerd={
        <>
          This is real <b>gradient descent</b> running in your browser: forward
          pass → cross-entropy loss → backprop through a scalar autograd engine
          → Adam update on all ~5,000 parameters. Loss starts at ln(27) ≈ 3.3
          (uniform guessing) and falls as the weights learn name statistics.
          One document per step, with linear learning-rate decay.
        </>
      }
    >
      <div className="rounded-lg border border-surface-border bg-surface p-6">
        <div className="mb-6 flex justify-center">
          <Tiny
            mood={tinyMood}
            say={
              phase === "idle"
                ? "My brain is full of random numbers... send me to school!"
                : phase === "training"
                  ? `Studying "${studying}"... 📖`
                  : "I graduated! My brain actually knows names now! 🎓"
            }
          />
        </div>

        {/* Confusion meter */}
        <div className="mx-auto mb-6 max-w-md">
          <div className="mb-1 flex justify-between text-xs">
            <span className="text-muted">
              confusion meter {phase !== "idle" && `(step ${step})`}
            </span>
            <span className={confusion > 0.5 ? "font-bold text-red-400" : "font-bold text-green"}>
              {phase === "idle" ? "100" : confusionPct}% confused{" "}
              {confusion > 0.7 ? "🥴" : confusion > 0.4 ? "🤔" : confusion > 0.15 ? "🙂" : "😎"}
            </span>
          </div>
          <div className="h-5 overflow-hidden rounded-full border border-surface-border bg-surface-light">
            <motion.div
              className="h-full rounded-full"
              style={{
                background: "linear-gradient(90deg, #22c55e, #f59e0b, #f87171)",
              }}
              animate={{ width: `${phase === "idle" ? 100 : confusionPct}%` }}
              transition={{ type: "spring", stiffness: 60, damping: 20 }}
            />
          </div>
          {mode === "nerd" && phase !== "idle" && (
            <p className="text-muted mt-1 text-right text-[10px]">
              loss ≈ {loss.toFixed(3)} (smoothed)
            </p>
          )}
        </div>

        <AnimatePresence mode="wait">
          {phase === "idle" && (
            <motion.div key="idle" exit={{ opacity: 0, scale: 0.9 }} className="text-center">
              <motion.button
                onClick={startSchool}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                animate={{ boxShadow: ["0 0 0px #f59e0b00", "0 0 24px #f59e0b55", "0 0 0px #f59e0b00"] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="rounded-xl border-2 border-amber bg-amber/10 px-8 py-4 text-lg font-bold text-amber"
              >
                🎒 Send Tiny to school!
              </motion.button>
              <p className="text-muted mt-3 text-xs">
                (~10 seconds of real training, right here in your browser)
              </p>
            </motion.div>
          )}

          {phase === "training" && (
            <motion.div
              key="training"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center"
            >
              <div className="flex items-center justify-center gap-1 text-2xl">
                {["📖", "✏️", "🧠", "💡"].map((e, i) => (
                  <motion.span
                    key={i}
                    animate={{ y: [0, -10, 0], opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.2 }}
                  >
                    {e}
                  </motion.span>
                ))}
              </div>
            </motion.div>
          )}

          {phase === "done" && (
            <motion.div key="done" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
              {beforeNames.length > 0 && (
                <div className="mb-6 grid grid-cols-2 gap-4">
                  <NameColumn
                    title="Before school 🥴"
                    names={beforeNames}
                    color="red"
                  />
                  <NameColumn
                    title="After school 🎓"
                    names={afterNames}
                    color="green"
                    delayStart={0.6}
                  />
                </div>
              )}
              <p className="text-muted mb-4 text-center text-xs">
                {mode === "simple"
                  ? "Same robot, same brain — it just practiced! These names are invented by Tiny, not copied."
                  : `${step} optimizer steps. The "after" samples reflect learned bigram/trigram structure — pronounceable, name-like, mostly novel.`}
              </p>
              <div className="text-center">
                <button
                  onClick={onDone}
                  className="rounded border border-green/40 bg-green/10 px-5 py-2 text-sm font-bold text-green transition-colors hover:bg-green/20"
                >
                  Time for a rematch →
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </ChapterShell>
  );
}

function NameColumn({
  title,
  names,
  color,
  delayStart = 0,
}: {
  title: string;
  names: string[];
  color: "red" | "green";
  delayStart?: number;
}) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        color === "red" ? "border-red-400/30 bg-red-400/5" : "border-green/30 bg-green/5"
      }`}
    >
      <p className="text-muted mb-2 text-center text-[10px] font-bold uppercase tracking-wider">
        {title}
      </p>
      <div className="space-y-1.5">
        {names.map((n, i) => (
          <motion.p
            key={i}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: delayStart + i * 0.15 }}
            className={`text-center font-bold ${color === "red" ? "text-red-400/80" : "text-green"}`}
          >
            {n}
          </motion.p>
        ))}
      </div>
    </div>
  );
}
