"use client";

import { motion } from "motion/react";

export type TinyMood =
  | "happy"
  | "excited"
  | "thinking"
  | "confused"
  | "sleepy"
  | "studying";

type Props = {
  mood?: TinyMood;
  size?: number;
  /** Optional speech bubble text */
  say?: string;
};

const MOOD_COLORS: Record<TinyMood, string> = {
  happy: "#22c55e",
  excited: "#f59e0b",
  thinking: "#60a5fa",
  confused: "#f87171",
  sleepy: "#a78bfa",
  studying: "#f59e0b",
};

/** Tiny — the animated robot mascot for Learn mode. Pure SVG + Motion. */
export default function Tiny({ mood = "happy", size = 120, say }: Props) {
  const accent = MOOD_COLORS[mood];
  const isExcited = mood === "excited";
  const isSleepy = mood === "sleepy";
  const isThinking = mood === "thinking" || mood === "studying";

  return (
    <div className="flex flex-col items-center gap-2">
      {say && (
        <motion.div
          key={say}
          initial={{ opacity: 0, y: 8, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          className="relative max-w-xs rounded-xl border border-surface-border bg-surface px-4 py-2 text-center text-sm"
        >
          {say}
          <div className="absolute -bottom-1.5 left-1/2 h-3 w-3 -translate-x-1/2 rotate-45 border-r border-b border-surface-border bg-surface" />
        </motion.div>
      )}

      <motion.div
        animate={
          isExcited
            ? { y: [0, -14, 0], rotate: [0, -4, 4, 0] }
            : { y: [0, -6, 0] }
        }
        transition={{
          duration: isExcited ? 0.7 : 2.4,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        <svg width={size} height={size} viewBox="0 0 120 120">
          {/* Antenna */}
          <line x1="60" y1="22" x2="60" y2="10" stroke={accent} strokeWidth="3" />
          <motion.circle
            cx="60"
            cy="8"
            r="5"
            fill={accent}
            animate={{ opacity: [1, 0.3, 1], scale: isThinking ? [1, 1.4, 1] : 1 }}
            transition={{ duration: 1, repeat: Infinity }}
          />

          {/* Head */}
          <rect x="28" y="22" width="64" height="50" rx="12" fill="#1e293b" stroke={accent} strokeWidth="2.5" />

          {/* Eyes */}
          {isSleepy ? (
            <>
              <line x1="40" y1="44" x2="52" y2="44" stroke={accent} strokeWidth="3" strokeLinecap="round" />
              <line x1="68" y1="44" x2="80" y2="44" stroke={accent} strokeWidth="3" strokeLinecap="round" />
            </>
          ) : (
            <>
              <motion.circle
                cx="46"
                cy="44"
                r={mood === "confused" ? 4 : 6}
                fill={accent}
                animate={{ scaleY: [1, 1, 0.1, 1] }}
                transition={{ duration: 3.5, repeat: Infinity, times: [0, 0.92, 0.96, 1] }}
              />
              <motion.circle
                cx="74"
                cy="44"
                r={mood === "confused" ? 7 : 6}
                fill={accent}
                animate={{ scaleY: [1, 1, 0.1, 1] }}
                transition={{ duration: 3.5, repeat: Infinity, times: [0, 0.92, 0.96, 1] }}
              />
            </>
          )}

          {/* Mouth */}
          {mood === "happy" || mood === "excited" ? (
            <path d="M 46 58 Q 60 68 74 58" stroke={accent} strokeWidth="3" fill="none" strokeLinecap="round" />
          ) : mood === "confused" ? (
            <path d="M 46 62 Q 53 56 60 62 Q 67 68 74 62" stroke={accent} strokeWidth="2.5" fill="none" strokeLinecap="round" />
          ) : isSleepy ? (
            <ellipse cx="60" cy="61" rx="6" ry="4" fill={accent} opacity={0.7} />
          ) : (
            <line x1="50" y1="61" x2="70" y2="61" stroke={accent} strokeWidth="3" strokeLinecap="round" />
          )}

          {/* Body */}
          <rect x="36" y="76" width="48" height="32" rx="8" fill="#1e293b" stroke={accent} strokeWidth="2.5" />
          {/* Chest light */}
          <motion.rect
            x="52"
            y="84"
            width="16"
            height="10"
            rx="3"
            fill={accent}
            animate={{ opacity: isThinking ? [0.3, 1, 0.3] : 0.8 }}
            transition={{ duration: 0.8, repeat: Infinity }}
          />

          {/* Arms */}
          <motion.line
            x1="36"
            y1="84"
            x2="22"
            y2={isExcited ? 70 : 94}
            stroke={accent}
            strokeWidth="3.5"
            strokeLinecap="round"
            animate={isExcited ? { y2: [70, 64, 70] } : undefined}
            transition={{ duration: 0.4, repeat: Infinity }}
          />
          <motion.line
            x1="84"
            y1="84"
            x2="98"
            y2={isExcited ? 70 : 94}
            stroke={accent}
            strokeWidth="3.5"
            strokeLinecap="round"
            animate={isExcited ? { y2: [70, 64, 70] } : undefined}
            transition={{ duration: 0.4, repeat: Infinity, delay: 0.2 }}
          />

          {/* Thinking dots */}
          {isThinking && (
            <g>
              {[0, 1, 2].map((i) => (
                <motion.circle
                  key={i}
                  cx={96 + i * 8}
                  cy={20 - i * 6}
                  r={2 + i}
                  fill={accent}
                  animate={{ opacity: [0, 1, 0] }}
                  transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.3 }}
                />
              ))}
            </g>
          )}

          {/* Zzz for sleepy */}
          {isSleepy && (
            <motion.text
              x="92"
              y="20"
              fill={accent}
              fontSize="14"
              fontWeight="bold"
              animate={{ opacity: [0, 1, 0], y: [24, 14] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              z z
            </motion.text>
          )}
        </svg>
      </motion.div>
    </div>
  );
}
