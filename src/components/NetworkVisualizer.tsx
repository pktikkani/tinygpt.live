"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "motion/react";

type Props = {
  totalParams: number;
  config: {
    nEmbd: number;
    nHead: number;
    nLayer: number;
    blockSize: number;
    vocabSize: number;
  };
  trainStep: number;
  trainDoc: string;
  trainLoss: number | null;
  encode: (s: string) => number[];
};

type BlockDef = {
  id: string;
  label: string;
  color: string;
  icon: string;
  params: number;
};

export default function NetworkVisualizer({
  totalParams,
  config,
  trainStep,
  trainDoc,
  trainLoss,
  encode,
}: Props) {
  const [expandedBlock, setExpandedBlock] = useState<string | null>(null);
  const [activeBlockIdx, setActiveBlockIdx] = useState(-1);
  const [displayWord, setDisplayWord] = useState("");
  const prevStepRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { nEmbd, nHead, vocabSize, blockSize } = config;

  const blocks: BlockDef[] = [
    {
      id: "input",
      label: "Input Token",
      color: "#f59e0b",
      icon: "Aa",
      params: 0,
    },
    {
      id: "embed",
      label: "Embedding",
      color: "#f59e0b",
      icon: "[ ]",
      params: vocabSize * nEmbd + blockSize * nEmbd,
    },
    {
      id: "attention",
      label: "Attention",
      color: "#22c55e",
      icon: "{}",
      params: 4 * nEmbd * nEmbd,
    },
    {
      id: "mlp",
      label: "MLP",
      color: "#3b82f6",
      icon: "fn",
      params: 2 * 4 * nEmbd * nEmbd,
    },
    {
      id: "output",
      label: "Output",
      color: "#a855f7",
      icon: "→",
      params: vocabSize * nEmbd,
    },
  ];

  // Compute word-specific info for each block
  const wordInfo = useMemo(() => {
    const word = displayWord || trainDoc;
    if (!word) return null;

    const tokens = encode(word);
    const chars = word.split("");
    const tokenPairs = chars.map((ch, i) => `${ch}→${tokens[i + 1]}`);
    const seqLen = tokens.length - 1; // excluding trailing BOS for context
    const headDim = nEmbd / nHead;
    const attentionPairs = (seqLen * (seqLen + 1)) / 2; // causal mask: each position attends to itself + prior

    return {
      input: {
        summary: tokenPairs.join(", "),
        detail: `"${word}" is split into ${chars.length} characters. Each is mapped to a token ID: ${tokenPairs.join(", ")}. A ^ (BOS=${tokens[0]}) marker is added at the start and end → [${tokens.join(", ")}] (${tokens.length} tokens total).`,
      },
      embed: {
        summary: `${chars.length} chars → ${chars.length}×${nEmbd} matrix`,
        detail: `Each of the ${chars.length} token IDs looks up a row in the ${vocabSize}×${nEmbd} embedding table, producing a ${nEmbd}-dimensional vector. Then position embeddings (0..${chars.length - 1}) from the ${blockSize}×${nEmbd} position table are added. Result: a ${chars.length}×${nEmbd} matrix — "${word}" is now ${chars.length * nEmbd} numbers.`,
      },
      attention: {
        summary: `${nHead} heads × ${attentionPairs} attention pairs`,
        detail: `Each of the ${nHead} attention heads (dim ${headDim}) computes Q, K, V projections for all ${seqLen} positions. With causal masking, each letter only sees itself and earlier letters — that's ${attentionPairs} total attention weights. For "${word}": "${chars[chars.length - 1]}" at position ${chars.length - 1} attends to all ${chars.length} characters, while "${chars[0]}" at position 0 only sees itself.`,
      },
      mlp: {
        summary: `${nEmbd}→${4 * nEmbd}→${nEmbd} per position`,
        detail: `Each of the ${seqLen} positions passes through a 2-layer network: ${nEmbd} inputs → ${4 * nEmbd} hidden neurons (ReLU) → ${nEmbd} outputs. For "${word}", that's ${seqLen} independent passes. This is where the model "thinks": given what attention gathered about the context of "${chars[chars.length - 1]}", what letter is likely next?`,
      },
      output: {
        summary: `${seqLen} positions → ${vocabSize} scores each`,
        detail: `The final linear layer maps each position's ${nEmbd}-dim vector to ${vocabSize} scores (one per possible next character: a-z + ^). For "${word}", the model produces ${seqLen}×${vocabSize} = ${seqLen * vocabSize} raw scores. The last position predicts what comes after "${chars[chars.length - 1]}" — training adjusts weights to make the correct next letter score highest.`,
      },
    };
  }, [displayWord, trainDoc, encode, nEmbd, nHead, vocabSize, blockSize]);

  // Short description for each block — dynamic when word is available
  function getDescription(blockId: string): string {
    if (!wordInfo) {
      const fallbacks: Record<string, string> = {
        input: "A letter comes in",
        embed: "Letter becomes a list of numbers",
        attention: "Letters look at each other",
        mlp: "Makes a decision",
        output: "Predicts the next letter",
      };
      return fallbacks[blockId] ?? "";
    }
    return wordInfo[blockId as keyof typeof wordInfo]?.summary ?? "";
  }

  function getDetail(blockId: string): string {
    if (!wordInfo) {
      const fallbacks: Record<string, string> = {
        input: `Each character (a-z) is converted to a number. There are ${vocabSize} possible tokens.`,
        embed: `Each token becomes a list of ${nEmbd} numbers. Position info is added.`,
        attention: `${nHead} heads examine relationships between letters.`,
        mlp: `Processes through ${nEmbd}→${4 * nEmbd}→${nEmbd} neurons.`,
        output: `Converts to ${vocabSize} scores — one per possible next character.`,
      };
      return fallbacks[blockId] ?? "";
    }
    return wordInfo[blockId as keyof typeof wordInfo]?.detail ?? "";
  }

  // Animate blocks sequentially on each train step
  useEffect(() => {
    if (trainStep <= prevStepRef.current) return;
    prevStepRef.current = trainStep;
    setDisplayWord(trainDoc);

    // Clear any pending animation
    if (timerRef.current) clearTimeout(timerRef.current);

    let idx = 0;
    const cycle = () => {
      setActiveBlockIdx(idx);
      idx++;
      if (idx < blocks.length) {
        timerRef.current = setTimeout(cycle, 350);
      } else {
        timerRef.current = setTimeout(() => setActiveBlockIdx(-1), 400);
      }
    };
    cycle();

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [trainStep, trainDoc, blocks.length]);

  const isActive = activeBlockIdx >= 0;

  return (
    <div className="rounded-lg border border-surface-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-amber text-sm font-bold tracking-wider uppercase">
          Architecture
        </h3>
        <span className="text-muted text-[10px]">
          {totalParams.toLocaleString()} params
        </span>
      </div>

      {/* Current word being processed */}
      <AnimatePresence mode="wait">
        {displayWord && (
          <motion.div
            key={displayWord + trainStep}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="mb-3 rounded bg-surface-light px-3 py-2 flex items-center justify-between"
          >
            <div className="flex items-center gap-2">
              <span className="text-muted text-[10px]">processing</span>
              <span className="text-amber font-bold text-sm tracking-wider">
                &quot;{displayWord}&quot;
              </span>
            </div>
            {trainLoss !== null && !isNaN(trainLoss) && (
              <span className={`text-[10px] font-bold ${trainLoss < 2 ? "text-green" : "text-amber"}`}>
                {trainLoss.toFixed(2)}
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Pipeline */}
      <div className="relative flex flex-col items-center">
        {blocks.map((block, i) => {
          const blockIsActive = activeBlockIdx === i;
          const blockIsPast = activeBlockIdx > i;

          return (
            <div key={block.id} className="flex flex-col items-center w-full">
              {/* Connection trace */}
              {i > 0 && (
                <div className="relative h-5 w-full flex justify-center">
                  {/* Trace line */}
                  <div
                    className="h-full w-px transition-all duration-150"
                    style={{
                      background: blockIsPast || blockIsActive
                        ? blocks[i - 1].color
                        : `${blocks[i - 1].color}20`,
                      boxShadow: blockIsActive
                        ? `0 0 6px ${blocks[i - 1].color}`
                        : "none",
                    }}
                  />
                  {/* Pulse dot */}
                  {blockIsActive && (
                    <motion.div
                      className="absolute left-1/2 -translate-x-1/2 h-1.5 w-1.5 rounded-full"
                      style={{
                        background: blocks[i - 1].color,
                        boxShadow: `0 0 8px ${blocks[i - 1].color}`,
                      }}
                      initial={{ top: 0 }}
                      animate={{ top: "100%" }}
                      transition={{ duration: 0.35, ease: "linear" }}
                    />
                  )}
                </div>
              )}

              {/* Block card */}
              <motion.button
                onClick={() =>
                  setExpandedBlock(expandedBlock === block.id ? null : block.id)
                }
                className="relative w-full rounded-lg border px-3 py-2 text-left cursor-pointer"
                style={{
                  borderColor: blockIsActive
                    ? block.color
                    : expandedBlock === block.id
                      ? `${block.color}60`
                      : `${block.color}20`,
                  background: blockIsActive
                    ? `${block.color}15`
                    : expandedBlock === block.id
                      ? `${block.color}0a`
                      : "transparent",
                  boxShadow: blockIsActive
                    ? `0 0 20px ${block.color}25, inset 0 0 20px ${block.color}08`
                    : "none",
                  transition: "border-color 0.15s, background 0.15s, box-shadow 0.15s",
                }}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                layout
              >
                <div className="flex items-center gap-2.5">
                  {/* Icon box */}
                  <div
                    className="relative flex h-7 w-7 shrink-0 items-center justify-center rounded text-[10px] font-bold"
                    style={{
                      background: blockIsActive ? `${block.color}30` : `${block.color}15`,
                      color: block.color,
                      border: `1px solid ${blockIsActive ? block.color : `${block.color}25`}`,
                      transition: "all 0.15s",
                    }}
                  >
                    {block.icon}
                    {blockIsActive && (
                      <motion.div
                        className="absolute inset-0 rounded"
                        style={{ border: `1.5px solid ${block.color}` }}
                        initial={{ scale: 1, opacity: 0.8 }}
                        animate={{ scale: 1.6, opacity: 0 }}
                        transition={{ duration: 0.4 }}
                      />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="text-xs font-bold"
                        style={{
                          color: block.color,
                          textShadow: blockIsActive ? `0 0 8px ${block.color}60` : "none",
                          transition: "text-shadow 0.15s",
                        }}
                      >
                        {block.label}
                      </span>
                      {block.params > 0 && (
                        <span className="text-muted text-[8px]">
                          {block.params.toLocaleString()}p
                        </span>
                      )}
                    </div>
                    <p className="text-muted text-[9px] leading-tight mt-0.5 truncate">
                      {getDescription(block.id)}
                    </p>
                  </div>

                  {/* Word flowing through this block */}
                  {blockIsActive && displayWord && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-[9px] font-bold shrink-0 rounded px-1.5 py-0.5"
                      style={{
                        color: block.color,
                        background: `${block.color}15`,
                        border: `1px solid ${block.color}30`,
                      }}
                    >
                      {displayWord}
                    </motion.span>
                  )}

                  {/* Expand arrow (only when not active) */}
                  {!blockIsActive && (
                    <motion.span
                      className="text-muted text-[9px] shrink-0"
                      animate={{ rotate: expandedBlock === block.id ? 90 : 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      ▶
                    </motion.span>
                  )}
                </div>

                {/* Expanded detail — word-specific */}
                <AnimatePresence>
                  {expandedBlock === block.id && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <p
                        className="mt-2 pt-2 text-[10px] leading-relaxed"
                        style={{
                          borderTop: `1px solid ${block.color}15`,
                          color: "var(--color-foreground)",
                          opacity: 0.75,
                        }}
                      >
                        {getDetail(block.id)}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="mt-3 pt-2 border-t border-surface-border">
        <p className="text-muted text-[9px] text-center">
          {isActive
            ? `"${displayWord}" flowing through the network...`
            : "Click any block to learn more. Train to see data flow."}
        </p>
      </div>
    </div>
  );
}
