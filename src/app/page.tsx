"use client";

import { useState, useMemo, useCallback } from "react";
import { motion } from "motion/react";
import Navigation from "@/components/Navigation";
import NetworkVisualizer from "@/components/NetworkVisualizer";
import AttentionHeatmap from "@/components/AttentionHeatmap";
import TemperatureControl from "@/components/TemperatureControl";
import TrainingStepper from "@/components/TrainingStepper";
import TokenFlow from "@/components/TokenFlow";
import { GPTModel } from "@/lib/gpt";
import type { AttentionData, TrainStepResult, GenerateResult } from "@/lib/gpt";

export default function Home() {
  const model = useMemo(() => new GPTModel(42), []);
  const arch = useMemo(() => model.getArchitectureInfo(), [model]);

  const [attentionData, setAttentionData] = useState<AttentionData[]>([]);
  const [tokenChars, setTokenChars] = useState<string[]>([]);
  const [trainStep, setTrainStep] = useState(0);
  const [trainDoc, setTrainDoc] = useState("");
  const [trainLoss, setTrainLoss] = useState<number | null>(null);

  const handleTrainStep = useCallback((result: TrainStepResult) => {
    setAttentionData(result.attentionData);
    const chars = ["<BOS>", ...result.doc.split(""), "<BOS>"];
    setTokenChars(chars);
    setTrainStep(result.step);
    setTrainDoc(result.doc);
    setTrainLoss(result.loss);
  }, []);

  const handleGenerate = useCallback((results: GenerateResult[]) => {
    if (results.length > 0 && results[0].attentionData.length > 0) {
      setAttentionData(results[0].attentionData);
      setTokenChars(["<BOS>", ...results[0].tokenChars]);
    }
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Hero */}
      <div className="pt-14">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-8"
          >
            <h1 className="text-amber glow-amber text-3xl font-bold tracking-wider mb-2">
              tinyGPT
            </h1>
            <p className="text-muted text-sm max-w-xl">
              A real GPT running in your browser. Port of Karpathy&apos;s
              pure-Python GPT to TypeScript. Train it, generate names, inspect
              attention heads, watch tokens flow.
            </p>

            <motion.a
              href="/learn"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-4 inline-flex items-center gap-2 rounded-lg border border-amber/40 bg-amber/10 px-4 py-2 text-sm font-bold text-amber transition-colors hover:bg-amber/20"
            >
              ✨ New to transformers? Take the animated tour — playful enough
              for kids, real math under the hood →
            </motion.a>

            {/* Stats bar */}
            <div className="mt-4 flex flex-wrap gap-4">
              {[
                { label: "params", value: arch.totalParams.toLocaleString() },
                { label: "heads", value: String(arch.config.nHead) },
                { label: "layers", value: String(arch.config.nLayer) },
                { label: "embed", value: String(arch.config.nEmbd) },
                { label: "vocab", value: String(arch.config.vocabSize) },
                { label: "ctx", value: String(arch.config.blockSize) },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="rounded border border-surface-border bg-surface px-3 py-1.5"
                >
                  <span className="text-muted text-[10px]">{label} </span>
                  <span className="text-green text-xs font-bold">{value}</span>
                </div>
              ))}
            </div>

            {/* Suggested flow */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="mt-4 flex flex-wrap items-center gap-x-2 gap-y-1 rounded border border-surface-border bg-surface px-3 py-2 text-xs"
            >
              <span className="text-amber font-bold">try this:</span>
              <span className="text-muted">
                <span className="text-green font-bold">①</span> Auto-train until
                loss &lt; 1.0
              </span>
              <span className="text-muted/50">→</span>
              <span className="text-muted">
                <span className="text-green font-bold">②</span> Generate names
              </span>
              <span className="text-muted/50">→</span>
              <span className="text-muted">
                <span className="text-green font-bold">③</span> Click H0–H3 in
                the Attention Heads panel to see what each head learned
              </span>
            </motion.div>
          </motion.div>

          {/* Main grid */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
            {/* Left column — Architecture */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="lg:col-span-3"
            >
              <NetworkVisualizer
                totalParams={arch.totalParams}
                config={arch.config}
                trainStep={trainStep}
                trainDoc={trainDoc}
                trainLoss={trainLoss}
                encode={model.tokenizer.encode}
              />
            </motion.div>

            {/* Center column — Training + Token Flow */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="space-y-4 lg:col-span-5"
            >
              <TrainingStepper model={model} onStep={handleTrainStep} />
              <TokenFlow model={model} />
            </motion.div>

            {/* Right column — Temperature + Attention */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="space-y-4 lg:col-span-4"
            >
              <TemperatureControl
                model={model}
                onGenerate={handleGenerate}
              />
              <AttentionHeatmap
                attentionData={attentionData}
                tokenChars={tokenChars}
                nHead={arch.config.nHead}
              />
            </motion.div>
          </div>

          {/* tinyGPT vs GPT-4 */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-12"
          >
            <h3 className="text-amber mb-3 text-sm font-bold tracking-wider uppercase">
              Same algorithm as the big ones
            </h3>
            <div className="overflow-x-auto rounded-lg border border-surface-border bg-surface">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-surface-border text-muted">
                    <th className="px-4 py-2.5 font-normal"></th>
                    <th className="text-green px-4 py-2.5 font-bold">tinyGPT</th>
                    <th className="text-amber px-4 py-2.5 font-bold">GPT-4</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Parameters", "~5,000", "~1,800,000,000,000"],
                    ["Training data", "200 names", "The internet"],
                    ["Generates", "Single names", "Essays, code, poetry"],
                    ["Runs on", "Your browser", "Massive GPU clusters"],
                  ].map(([label, tiny, big]) => (
                    <tr key={label} className="border-b border-surface-border last:border-0">
                      <td className="text-muted px-4 py-2.5">{label}</td>
                      <td className="text-green px-4 py-2.5 font-medium">{tiny}</td>
                      <td className="px-4 py-2.5">{big}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-muted mt-2 text-[11px]">
              Same architecture — embeddings, multi-head attention, MLP,
              softmax. Just a few orders of magnitude smaller.
            </p>
          </motion.div>

          {/* Footer */}
          <div className="mt-12 border-t border-surface-border pt-6 pb-8 text-center">
            <p className="text-muted text-xs">
              Built with Next.js 16 + React 19 + TypeScript + Tailwind v4 +
              Motion 12 + Recharts
            </p>
            <p className="text-muted/50 mt-1 text-[10px]">
              Based on{" "}
              <a
                href="https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95"
                target="_blank"
                rel="noopener noreferrer"
                className="text-amber/70 underline decoration-dotted underline-offset-2 transition-colors hover:text-amber"
              >
                Karpathy&apos;s pure-Python GPT
              </a>{" "}
              — the most atomic way to train a transformer
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
