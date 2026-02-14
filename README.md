# tinyGPT.live

A real GPT running entirely in your browser. No backend, no API keys — just a transformer learning to generate names in real time.

Built as an interactive visualization of a pure-Python GPT, ported to TypeScript. Watch tokens flow through attention heads, step through training, inspect what the model learns, and generate names with adjustable temperature.

## What You Can Do

- **Train step-by-step** — Click "Step" and watch the loss drop from ~3.3 (random guessing) to under 1.0 (actually learned). Set a target loss and let it auto-train.
- **Generate names** — Slide the temperature from 0.05 (conservative) to 1.5 (wild) and generate character-level names.
- **Watch tokens flow** — Type a name and see it pass through Input → Embedding → Attention → MLP → Output in real time.
- **Inspect attention heads** — Click H0-H3 to see which letters are paying attention to which. Bright = high attention.
- **Understand the architecture** — Each layer shows what's actually happening to the word being processed, with real token IDs, matrix dimensions, and attention pair counts.

## The Model

| | tinyGPT | GPT-4 |
|---|---|---|
| Parameters | ~5,000 | ~1,800,000,000,000 |
| Training data | 200 names | The internet |
| Generates | Single names | Essays, code, poetry |
| Runs on | Your browser | Massive GPU clusters |

Same algorithm. Same architecture. Just a few orders of magnitude smaller.

## Tech Stack

- Next.js 16 + React 19
- TypeScript (full GPT engine, no dependencies)
- Tailwind CSS v4
- Motion 12 (animations)
- Recharts (loss chart)

## Run Locally

```bash
git clone https://github.com/pktikkani/tinygpt.live.git
cd tinygpt.live
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## How It Works

The entire GPT — tokenizer, embeddings, multi-head attention, MLP, softmax, cross-entropy loss, backpropagation with autograd, Adam optimizer — runs as scalar TypeScript in the browser. No WebGL, no WASM, no workers. Just math.

The model trains on ~200 common names (emma, oliver, sophia...) and learns character-level patterns: names often start with vowels, "th" goes together, names end with "a" or "n". After enough training, it generates surprisingly plausible new names.

## License

MIT
