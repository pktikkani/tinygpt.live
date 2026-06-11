"use client";

import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { GPTModel } from "@/lib/gpt";
import { NAMES } from "@/lib/names-data";
import type { LearnMode } from "./ChapterShell";
import ChapterShell from "./ChapterShell";
import Tiny, { type TinyMood } from "./Tiny";

type Props = {
  model: GPTModel;
  mode: LearnMode;
  variant: "untrained" | "trained";
  onDone: (human: number, tiny: number) => void;
};

const ALPHABET = "abcdefghijklmnopqrstuvwxyz".split("");
const START_REVEALED = 2;

function pickName(): string {
  const candidates = NAMES.filter((n) => n.length >= 5 && n.length <= 8);
  return candidates[Math.floor(Math.random() * candidates.length)];
}

export default function GuessGame({ model, mode, variant, onDone }: Props) {
  const [name, setName] = useState(pickName);
  const [revealed, setRevealed] = useState(START_REVEALED);
  const [humanScore, setHumanScore] = useState(0);
  const [tinyScore, setTinyScore] = useState(0);
  const [round, setRound] = useState<{
    humanGuess: string;
    tinyGuess: string;
    tinyProb: number;
    actual: string;
  } | null>(null);
  const [finished, setFinished] = useState(false);

  const totalRounds = name.length - START_REVEALED;

  const handleGuess = useCallback(
    (letter: string) => {
      if (round || finished) return;
      const prefix = name.slice(0, revealed);
      const { topTokens } = model.predictNext(prefix);
      const best = topTokens.find((t) => t.char !== "<BOS>")!;
      const actual = name[revealed];

      setRound({ humanGuess: letter, tinyGuess: best.char, tinyProb: best.prob, actual });
      if (letter === actual) setHumanScore((s) => s + 1);
      if (best.char === actual) setTinyScore((s) => s + 1);
    },
    [round, finished, name, revealed, model]
  );

  const nextRound = useCallback(() => {
    const newRevealed = revealed + 1;
    setRound(null);
    setRevealed(newRevealed);
    if (newRevealed >= name.length) {
      setFinished(true);
    }
  }, [revealed, name]);

  const playAgain = useCallback(() => {
    setName(pickName());
    setRevealed(START_REVEALED);
    setHumanScore(0);
    setTinyScore(0);
    setRound(null);
    setFinished(false);
  }, []);

  const tinyMood: TinyMood = finished
    ? tinyScore >= humanScore
      ? "excited"
      : "confused"
    : round
      ? round.tinyGuess === round.actual
        ? "happy"
        : "confused"
      : "thinking";

  const tinySpeech = finished
    ? variant === "untrained"
      ? tinyScore < humanScore
        ? "I lost... I was just guessing randomly. I haven't been to school yet! 😅"
        : "Lucky me! But honestly, I was guessing randomly..."
      : tinyScore >= humanScore
        ? "School paid off! I learned the patterns in names! 🎓"
        : "You're good! But did you see how close my guesses were?"
    : round
      ? round.tinyGuess === round.actual
        ? `Yes! I guessed "${round.tinyGuess}"!`
        : `Hmm, I guessed "${round.tinyGuess}"... wrong!`
      : variant === "untrained"
        ? "I'll guess too! (no promises, my brain is full of random numbers)"
        : "Rematch! My brain has been trained now. Bring it on!";

  const isUntrained = variant === "untrained";

  return (
    <ChapterShell
      emoji={isUntrained ? "🎮" : "🏆"}
      title={isUntrained ? "The Guessing Game" : "The Rematch!"}
      mode={mode}
      simple={
        isUntrained ? (
          <>
            This is the ONLY game Tiny ever plays: <b>guess the next letter</b>.
            That&apos;s really all ChatGPT does too — over and over, very fast!
            Below is a real name with some letters hidden. You and Tiny both
            guess the next letter. Who&apos;s better?
          </>
        ) : (
          <>
            Tiny went to school and studied hundreds of names! Now play the same
            game again. Watch how much better Tiny&apos;s guesses are — it
            learned which letters like to follow each other.
          </>
        )
      }
      nerd={
        isUntrained ? (
          <>
            Next-token prediction: given prefix <code>x₁..xₜ</code>, output{" "}
            <code>P(xₜ₊₁)</code> over the vocab. The model is freshly
            initialized (Gaussian, σ=0.08), so its softmax output is near
            uniform — ~1/27 ≈ 3.7% per token. You have priors about English
            names; it has none. Yet.
          </>
        ) : (
          <>
            Same forward pass, but now the weights encode real statistics:
            vowel/consonant alternation, common endings (-er, -on, -lyn).
            Watch the confidence percentage — a trained model puts high
            probability mass on few tokens instead of spreading it uniformly.
          </>
        )
      }
    >
      <div className="rounded-lg border border-surface-border bg-surface p-6">
        {/* Scoreboard */}
        <div className="mb-6 flex items-center justify-center gap-8">
          <ScoreCard label="You" score={humanScore} color="amber" />
          <span className="text-muted text-lg font-bold">vs</span>
          <ScoreCard label="Tiny" score={tinyScore} color="green" />
        </div>

        <div className="mb-6 flex justify-center">
          <Tiny mood={tinyMood} size={100} say={tinySpeech} />
        </div>

        {/* The name being guessed */}
        <div className="mb-6 flex flex-wrap items-center justify-center gap-2">
          {name.split("").map((ch, i) => {
            const isRevealed = i < revealed;
            const isNext = i === revealed && !finished;
            return (
              <motion.div
                key={i}
                animate={
                  isNext
                    ? { scale: [1, 1.08, 1], borderColor: ["#f59e0b66", "#f59e0b", "#f59e0b66"] }
                    : {}
                }
                transition={{ duration: 1.2, repeat: isNext ? Infinity : 0 }}
                className={`flex h-14 w-12 items-center justify-center rounded-lg border-2 text-2xl font-bold ${
                  isRevealed
                    ? "border-green/40 bg-green/10 text-green"
                    : isNext
                      ? "border-amber bg-amber/10 text-amber"
                      : "border-surface-border bg-surface-light text-muted"
                }`}
              >
                {isRevealed ? (
                  <motion.span
                    initial={{ rotateY: 90, opacity: 0 }}
                    animate={{ rotateY: 0, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 200 }}
                  >
                    {ch}
                  </motion.span>
                ) : isNext ? (
                  "?"
                ) : (
                  "·"
                )}
              </motion.div>
            );
          })}
        </div>

        <AnimatePresence mode="wait">
          {finished ? (
            <motion.div
              key="done"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center"
            >
              <p className="mb-1 text-lg font-bold">
                {humanScore > tinyScore ? (
                  <span className="text-amber glow-amber">🎉 You win {humanScore}–{tinyScore}!</span>
                ) : humanScore < tinyScore ? (
                  <span className="text-green glow-green">🤖 Tiny wins {tinyScore}–{humanScore}!</span>
                ) : (
                  <span className="text-muted">🤝 Tie game, {humanScore}–{tinyScore}!</span>
                )}
              </p>
              <p className="text-muted mb-4 text-xs">the name was &quot;{name}&quot;</p>
              <div className="flex justify-center gap-3">
                <button
                  onClick={playAgain}
                  className="rounded border border-surface-border bg-surface-light px-4 py-2 text-sm transition-colors hover:border-amber/50"
                >
                  Play again
                </button>
                <button
                  onClick={() => onDone(humanScore, tinyScore)}
                  className="rounded border border-green/40 bg-green/10 px-4 py-2 text-sm font-bold text-green transition-colors hover:bg-green/20"
                >
                  Continue →
                </button>
              </div>
            </motion.div>
          ) : round ? (
            <motion.div
              key="reveal"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              <div className="mb-4 flex items-center justify-center gap-4">
                <GuessChip label="You" guess={round.humanGuess} correct={round.humanGuess === round.actual} delay={0} />
                <GuessChip
                  label="Tiny"
                  guess={round.tinyGuess}
                  correct={round.tinyGuess === round.actual}
                  delay={0.5}
                  sub={`${(round.tinyProb * 100).toFixed(0)}% sure`}
                />
                <motion.div
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.1, type: "spring" }}
                  className="flex flex-col items-center"
                >
                  <span className="text-muted text-[10px] uppercase">Answer</span>
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg border-2 border-green bg-green/20 text-xl font-bold text-green">
                    {round.actual}
                  </div>
                </motion.div>
              </div>
              <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.4 }}
                onClick={nextRound}
                className="rounded border border-amber/40 bg-amber/10 px-5 py-2 text-sm font-bold text-amber transition-colors hover:bg-amber/20"
              >
                {revealed + 1 >= name.length ? "See results →" : "Next letter →"}
              </motion.button>
            </motion.div>
          ) : (
            <motion.div key="pick" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <p className="text-muted mb-3 text-center text-xs">
                Round {revealed - START_REVEALED + 1} of {totalRounds} — pick the next letter:
              </p>
              <div className="mx-auto flex max-w-md flex-wrap justify-center gap-1.5">
                {ALPHABET.map((letter, i) => (
                  <motion.button
                    key={letter}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.015 }}
                    whileHover={{ scale: 1.15, y: -3 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => handleGuess(letter)}
                    className="h-10 w-10 rounded-lg border border-surface-border bg-surface-light text-sm font-bold transition-colors hover:border-amber hover:text-amber"
                  >
                    {letter}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </ChapterShell>
  );
}

function ScoreCard({ label, score, color }: { label: string; score: number; color: "amber" | "green" }) {
  return (
    <div className="text-center">
      <p className="text-muted text-[10px] uppercase tracking-wider">{label}</p>
      <motion.p
        key={score}
        initial={{ scale: 1.6 }}
        animate={{ scale: 1 }}
        className={`text-3xl font-bold ${color === "amber" ? "text-amber glow-amber" : "text-green glow-green"}`}
      >
        {score}
      </motion.p>
    </div>
  );
}

function GuessChip({
  label,
  guess,
  correct,
  delay,
  sub,
}: {
  label: string;
  guess: string;
  correct: boolean;
  delay: number;
  sub?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, type: "spring" }}
      className="flex flex-col items-center"
    >
      <span className="text-muted text-[10px] uppercase">{label}</span>
      <motion.div
        animate={correct ? { rotate: [0, -8, 8, 0] } : {}}
        transition={{ delay: delay + 1.2 }}
        className={`flex h-12 w-12 items-center justify-center rounded-lg border-2 text-xl font-bold ${
          correct ? "border-green bg-green/20 text-green" : "border-red-400/60 bg-red-400/10 text-red-400"
        }`}
      >
        {guess}
      </motion.div>
      <span className="text-muted mt-0.5 text-[9px]">{sub ?? (correct ? "✓ +1" : "✗")}</span>
    </motion.div>
  );
}
