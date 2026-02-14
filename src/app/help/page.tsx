"use client";

import { motion } from "motion/react";
import Navigation from "@/components/Navigation";

const sections = [
  {
    id: "big-picture",
    title: "The Big Picture",
    content: `You know how autocomplete on your phone suggests the next word when you type? This app shows you exactly how that works — but for names instead of sentences.

This is a tiny AI brain that learns to make up new names. It starts completely clueless (like a newborn), and you teach it by showing it real names like "emma", "oliver", "sophia". After seeing enough examples, it figures out the patterns — names often start with vowels, "th" goes together, names end with "a" or "n" a lot — and starts inventing its own names that sound real.

ChatGPT, Claude, and other AI chatbots work on the exact same principle, just millions of times bigger. This little model has about 5,000 "brain cells" (parameters). ChatGPT has hundreds of billions. But the core idea is identical: predict what comes next.`,
  },
  {
    id: "what-are-tokens",
    title: "What Are Tokens?",
    content: `Before the AI can read anything, it needs to convert letters into numbers — because computers only understand numbers.

Think of it like a simple code: a=0, b=1, c=2, ... z=25. There's also a special "start/stop" symbol (shown as ^) that tells the model "a name begins here" or "a name ends here."

So the name "emma" becomes the numbers: [^, 4, 12, 12, 0, ^]

That's all a tokenizer does — it's a translator between human letters and computer numbers. Real AI models like ChatGPT use a fancier version that converts whole words or word pieces into numbers, but the idea is the same.`,
  },
  {
    id: "how-it-learns",
    title: "How Does It Learn?",
    content: `Imagine you're learning a new language by reading thousands of names. After a while, you'd notice patterns — you wouldn't see "zxq" in a name, but "tion" is common.

The AI learns the same way, but with math:

1. You show it a name like "emma"
2. It tries to guess each next letter: after "e", what comes next? It might guess "z" (wrong!)
3. You tell it the right answer was "m"
4. It adjusts its brain slightly to make "m" more likely next time it sees "e" in this position
5. Repeat thousands of times

The "loss" number you see during training measures how wrong the guesses are. High loss (3.0+) means "clueless, just guessing randomly." Low loss (under 2.0) means "getting pretty good at this." You literally watch it go from dumb to smart.`,
  },
  {
    id: "what-is-attention",
    title: "What Is Attention?",
    content: `This is the secret sauce that made modern AI possible. It was invented in 2017 and changed everything.

When the model is trying to figure out the next letter, it needs to look back at previous letters. But not all previous letters matter equally. Attention is how the model decides which letters to focus on.

For example, when generating a name and the model has seen "mar" so far:
- It might pay a lot of attention to "m" (names starting with "mar" often continue with "y", "k", "ia")
- It might ignore "a" (less useful for the decision)

The app has 4 "attention heads" (H0, H1, H2, H3). Each head learns to look for different things — one might focus on "what was the last letter?", another on "what letter did the name start with?", another on "how long has this name been going?" They work together like a team.

The heatmap shows this visually: bright green = "I'm paying a lot of attention to that letter." Dark = "I'm ignoring it."`,
  },
  {
    id: "temperature",
    title: "What Is Temperature?",
    content: `Temperature controls how adventurous or cautious the AI is when picking the next letter.

Think of it like ordering food:
- Low temperature (0.1-0.3) = "I'll have my usual" — safe, predictable, common names
- Medium temperature (0.5) = "Let me try something on the menu I haven't had" — balanced
- High temperature (0.8-1.5) = "Surprise me, chef!" — wild, creative, sometimes weird names

At low temperature, the model always picks the letter it's most confident about. At high temperature, it's willing to take risks and pick less likely letters, which leads to more creative (and sometimes nonsensical) results.

This is the same "temperature" setting you see in ChatGPT and other AI tools. Now you know what it actually does!`,
  },
  {
    id: "architecture",
    title: "What's Inside the Brain?",
    content: `The left panel shows the layers the data passes through, like an assembly line:

1. Token Embedding — Converts each letter into a list of 16 numbers that represent its "meaning." The letter "a" might become [0.3, -0.1, 0.8, ...]. These numbers are learned during training.

2. Position Embedding — Adds information about where in the name each letter is. The model needs to know that "a" at position 1 is different from "a" at position 4.

3. Attention — The letters look at each other and share information (explained above).

4. MLP (Multi-Layer Perceptron) — A mini brain-within-the-brain that processes the information gathered by attention. Think of attention as "gathering clues" and MLP as "making a decision based on those clues."

5. LM Head — Takes the final processed information and converts it back into a prediction: "the next letter should be..." with a probability for each possible letter.

This tiny model does this once (1 layer). GPT-4 does it about 120 times in sequence, with each layer refining the prediction further.`,
  },
  {
    id: "using",
    title: "How to Use This App",
    content: `Start here! Follow these steps in order:

Step 1: Train the model
Click "Step" in the Training panel, or click "Auto" to train continuously. Watch the loss number — it starts around 3.3 (random guessing) and should drop below 2.0 (actually learned something). Train for at least 100 steps. You're literally watching an AI learn in real time.

Step 2: Generate names
Go to Temperature Control, set the slider to 0.5, and click "Generate Names." You'll see 8 made-up names. Try sliding temperature to 0.2 (boring but realistic) and then 1.0 (creative but weird) to feel the difference.

Step 3: Watch tokens flow
In Token Flow, type a name like "emma" and click "Flow." Watch the letters pass through each stage of the brain: Input → Embedding → Attention → MLP → Output. At the end, the model generates a continuation — what it thinks should come after those letters.

Step 4: Inspect attention
Look at the Attention Heatmap. Click H0, H1, H2, H3 to see what each attention head learned. The grid shows which letters are "looking at" which other letters. Bright green = paying attention, dark = ignoring.

Step 5: Experiment!
- Train to 1000+ steps for better names
- Try extreme temperatures (0.05 vs 1.5)
- Type different names in Token Flow and see how predictions change
- Compare attention patterns at step 50 vs step 500`,
  },
  {
    id: "so-what",
    title: "Why Does This Matter?",
    content: `Every AI chatbot — ChatGPT, Claude, Gemini, Llama — is built on this exact same idea, just scaled up massively:

- This model: 5,000 parameters, learns from 200 names, generates single names
- GPT-4: ~1,800,000,000,000 parameters, learned from the entire internet, generates essays, code, poetry

But the core algorithm is the same: tokenize the input, pass it through layers of attention and processing, predict the next piece. What you just played with IS the algorithm behind modern AI. Everything else is just making it bigger and faster.

The fact that this simple recipe — "predict the next token" — leads to something that can write poetry, solve math, and hold conversations is one of the most surprising discoveries in computer science. And now you've seen it with your own eyes.`,
  },
];

export default function HelpPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="pt-14">
        <div className="mx-auto max-w-3xl px-6 py-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-amber glow-amber text-2xl font-bold tracking-wider mb-2">
              HOW IT WORKS
            </h1>
            <p className="text-muted text-sm mb-8">
              No PhD required. A plain-English guide to how AI actually works,
              and how to use this app.
            </p>
          </motion.div>

          {/* Table of contents */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="mb-8 rounded-lg border border-surface-border bg-surface p-4"
          >
            <h3 className="text-green text-xs font-bold uppercase tracking-wider mb-2">
              Contents
            </h3>
            <div className="space-y-1">
              {sections.map((s, i) => (
                <a
                  key={s.id}
                  href={`#${s.id}`}
                  className="text-muted hover:text-amber flex items-center gap-2 text-sm transition-colors"
                >
                  <span className="text-green/50 text-[10px]">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {s.title}
                </a>
              ))}
            </div>
          </motion.div>

          {/* Sections */}
          <div className="space-y-6">
            {sections.map((section, i) => (
              <motion.section
                key={section.id}
                id={section.id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 + i * 0.05 }}
                className="rounded-lg border border-surface-border bg-surface p-5"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-green text-[10px] font-bold">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <h2 className="text-amber text-sm font-bold tracking-wider uppercase">
                    {section.title}
                  </h2>
                </div>
                <div className="text-foreground/80 text-sm leading-relaxed whitespace-pre-line">
                  {section.content}
                </div>
              </motion.section>
            ))}
          </div>

          {/* Credits */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-8 border-t border-surface-border pt-6 pb-8 text-center"
          >
            <p className="text-muted text-xs">
              Based on Andrej Karpathy&apos;s pure-Python GPT —
              &ldquo;The most atomic way to train and inference a GPT&rdquo;
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
