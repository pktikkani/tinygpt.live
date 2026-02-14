/**
 * GPT model — TypeScript port of Karpathy's pure-Python GPT.
 * Runs entirely in the browser: tokenizer, model, training, inference.
 *
 * Architecture: GPT-2 style with rmsnorm (no layernorm), no biases, ReLU (not GeLU).
 * Config: n_embd=16, n_head=4, n_layer=1, block_size=16, vocab_size=27 (a-z + BOS)
 */

import { Value, vsum } from "./value";
import { NAMES } from "./names-data";

// --- Seeded PRNG (deterministic like Python's random.seed(42)) ---
class SeededRandom {
  private s: number;
  constructor(seed: number) {
    this.s = seed;
  }
  next(): number {
    this.s = (this.s * 1664525 + 1013904223) & 0xffffffff;
    return (this.s >>> 0) / 0xffffffff;
  }
  gauss(mean: number, std: number): number {
    // Box-Muller transform
    const u1 = this.next();
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }
  shuffle<T>(arr: T[]): T[] {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  choices(weights: number[]): number {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = this.next() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }
}

// --- Tokenizer ---
export type Tokenizer = {
  uchars: string[];
  BOS: number;
  vocabSize: number;
  encode: (s: string) => number[];
  decode: (ids: number[]) => string;
  tokenToChar: (id: number) => string;
};

export function createTokenizer(docs: string[]): Tokenizer {
  const charSet = new Set<string>();
  for (const doc of docs) {
    for (const ch of doc) charSet.add(ch);
  }
  const uchars = [...charSet].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;

  return {
    uchars,
    BOS,
    vocabSize,
    encode: (s: string) => [BOS, ...s.split("").map((ch) => uchars.indexOf(ch)), BOS],
    decode: (ids: number[]) => ids.filter((id) => id !== BOS).map((id) => uchars[id]).join(""),
    tokenToChar: (id: number) => (id === BOS ? "<BOS>" : uchars[id] ?? "?"),
  };
}

// --- Model types ---
export type Matrix = Value[][];
export type StateDict = Record<string, Matrix>;

export type GPTConfig = {
  nEmbd: number;
  nHead: number;
  nLayer: number;
  blockSize: number;
  vocabSize: number;
};

export const DEFAULT_CONFIG: GPTConfig = {
  nEmbd: 16,
  nHead: 4,
  nLayer: 1,
  blockSize: 16,
  vocabSize: 0, // set by tokenizer
};

// Attention weights captured during forward pass for visualization
export type AttentionData = {
  layer: number;
  head: number;
  weights: number[][]; // [query_pos][key_pos] attention weights
};

export type ForwardResult = {
  logits: Value[];
  attentionData: AttentionData[];
};

export type TrainStepResult = {
  loss: number;
  step: number;
  doc: string;
  attentionData: AttentionData[];
};

export type GenerateResult = {
  text: string;
  tokens: number[];
  tokenChars: string[];
  attentionData: AttentionData[];
};

// --- GPT Model class ---
export class GPTModel {
  config: GPTConfig;
  stateDict: StateDict;
  params: Value[];
  tokenizer: Tokenizer;
  private rng: SeededRandom;
  private docs: string[];

  // Adam optimizer state
  private m: number[];
  private v: number[];
  private step: number;
  private learningRate: number;
  private beta1: number;
  private beta2: number;
  private epsAdam: number;

  constructor(seed = 42) {
    this.rng = new SeededRandom(seed);
    this.docs = this.rng.shuffle(NAMES);
    this.tokenizer = createTokenizer(this.docs);

    this.config = {
      ...DEFAULT_CONFIG,
      vocabSize: this.tokenizer.vocabSize,
    };

    const { nEmbd, nHead, nLayer, blockSize, vocabSize } = this.config;
    const headDim = nEmbd / nHead;

    // Initialize parameters
    const matrix = (nout: number, nin: number, std = 0.08): Matrix => {
      const m: Matrix = [];
      for (let i = 0; i < nout; i++) {
        const row: Value[] = [];
        for (let j = 0; j < nin; j++) {
          row.push(new Value(this.rng.gauss(0, std)));
        }
        m.push(row);
      }
      return m;
    };

    this.stateDict = {
      wte: matrix(vocabSize, nEmbd),
      wpe: matrix(blockSize, nEmbd),
      lm_head: matrix(vocabSize, nEmbd),
    };

    for (let i = 0; i < nLayer; i++) {
      this.stateDict[`layer${i}.attn_wq`] = matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.attn_wk`] = matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.attn_wv`] = matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.attn_wo`] = matrix(nEmbd, nEmbd);
      this.stateDict[`layer${i}.mlp_fc1`] = matrix(4 * nEmbd, nEmbd);
      this.stateDict[`layer${i}.mlp_fc2`] = matrix(nEmbd, 4 * nEmbd);
    }

    // Flatten params
    this.params = [];
    for (const key of Object.keys(this.stateDict)) {
      for (const row of this.stateDict[key]) {
        for (const p of row) {
          this.params.push(p);
        }
      }
    }

    // Adam state
    this.m = new Array(this.params.length).fill(0);
    this.v = new Array(this.params.length).fill(0);
    this.step = 0;
    this.learningRate = 0.01;
    this.beta1 = 0.85;
    this.beta2 = 0.99;
    this.epsAdam = 1e-8;
  }

  getParamCount(): number {
    return this.params.length;
  }

  getCurrentStep(): number {
    return this.step;
  }

  // --- Core math operations ---
  private linear(x: Value[], w: Matrix): Value[] {
    return w.map((wo) => vsum(wo.map((wi, i) => wi.mul(x[i]))));
  }

  private softmax(logits: Value[]): Value[] {
    const maxVal = Math.max(...logits.map((v) => v.data));
    const exps = logits.map((v) => v.sub(maxVal).exp());
    const total = vsum(exps);
    return exps.map((e) => e.div(total));
  }

  private rmsnorm(x: Value[]): Value[] {
    const ms = vsum(x.map((xi) => xi.mul(xi))).div(x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map((xi) => xi.mul(scale));
  }

  // Forward pass for a single token
  private forward(
    tokenId: number,
    posId: number,
    keys: Value[][][],
    values: Value[][][],
    collectAttention: boolean
  ): ForwardResult {
    const { nEmbd, nHead, nLayer } = this.config;
    const headDim = nEmbd / nHead;
    const attentionData: AttentionData[] = [];

    // Token + position embedding
    const tokEmb = this.stateDict.wte[tokenId];
    const posEmb = this.stateDict.wpe[posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = this.rmsnorm(x);

    for (let li = 0; li < nLayer; li++) {
      // 1) Multi-head attention
      const xResidual = x;
      x = this.rmsnorm(x);
      const q = this.linear(x, this.stateDict[`layer${li}.attn_wq`]);
      const k = this.linear(x, this.stateDict[`layer${li}.attn_wk`]);
      const v = this.linear(x, this.stateDict[`layer${li}.attn_wv`]);
      keys[li].push(k);
      values[li].push(v);

      const xAttn: Value[] = [];
      for (let h = 0; h < nHead; h++) {
        const hs = h * headDim;
        const qH = q.slice(hs, hs + headDim);
        const kH = keys[li].map((ki) => ki.slice(hs, hs + headDim));
        const vH = values[li].map((vi) => vi.slice(hs, hs + headDim));

        // Compute attention logits
        const attnLogits = kH.map((kt) =>
          vsum(qH.map((qj, j) => qj.mul(kt[j]))).div(Math.sqrt(headDim))
        );

        const attnWeights = this.softmax(attnLogits);

        // Capture attention weights for visualization
        if (collectAttention) {
          attentionData.push({
            layer: li,
            head: h,
            weights: [attnWeights.map((w) => w.data)],
          });
        }

        // Weighted sum of values
        for (let j = 0; j < headDim; j++) {
          const headOut = vsum(attnWeights.map((w, t) => w.mul(vH[t][j])));
          xAttn.push(headOut);
        }
      }

      x = this.linear(xAttn, this.stateDict[`layer${li}.attn_wo`]);
      x = x.map((a, i) => a.add(xResidual[i]));

      // 2) MLP block
      const xRes2 = x;
      x = this.rmsnorm(x);
      x = this.linear(x, this.stateDict[`layer${li}.mlp_fc1`]);
      x = x.map((xi) => xi.relu());
      x = this.linear(x, this.stateDict[`layer${li}.mlp_fc2`]);
      x = x.map((a, i) => a.add(xRes2[i]));
    }

    const logits = this.linear(x, this.stateDict.lm_head);
    return { logits, attentionData };
  }

  // --- Training step ---
  trainStep(numSteps = 1000): TrainStepResult {
    const doc = this.docs[this.step % this.docs.length];
    const tokens = this.tokenizer.encode(doc);
    const n = Math.min(this.config.blockSize, tokens.length - 1);

    // Forward pass
    const { nLayer } = this.config;
    const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
    const vals: Value[][][] = Array.from({ length: nLayer }, () => []);
    const losses: Value[] = [];
    let allAttention: AttentionData[] = [];

    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const { logits, attentionData } = this.forward(tokenId, posId, keys, vals, true);
      const probs = this.softmax(logits);
      const lossT = probs[targetId].log().neg();
      losses.push(lossT);

      // Merge attention data (accumulate per-head per-layer)
      for (const ad of attentionData) {
        const existing = allAttention.find(
          (a) => a.layer === ad.layer && a.head === ad.head
        );
        if (existing) {
          existing.weights.push(ad.weights[0]);
        } else {
          allAttention.push({ ...ad });
        }
      }
    }

    const loss = vsum(losses).div(n);

    // Backward
    loss.backward();

    // Adam update
    const lrT = this.learningRate * Math.max(0.01, 1 - this.step / numSteps);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * p.grad;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * p.grad ** 2;
      const mHat = this.m[i] / (1 - this.beta1 ** (this.step + 1));
      const vHat = this.v[i] / (1 - this.beta2 ** (this.step + 1));
      p.data -= lrT * mHat / (Math.sqrt(vHat) + this.epsAdam);
      p.grad = 0;
    }

    this.step++;

    return {
      loss: loss.data,
      step: this.step,
      doc,
      attentionData: allAttention,
    };
  }

  // --- Inference ---
  generate(temperature = 0.5, maxLen?: number): GenerateResult {
    const { nLayer, blockSize } = this.config;
    const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
    const vals: Value[][][] = Array.from({ length: nLayer }, () => []);
    let tokenId = this.tokenizer.BOS;
    const tokenIds: number[] = [];
    const tokenChars: string[] = [];
    let allAttention: AttentionData[] = [];
    const len = maxLen ?? blockSize;

    for (let posId = 0; posId < len; posId++) {
      const { logits, attentionData } = this.forward(tokenId, posId, keys, vals, true);

      // Apply temperature
      const scaledLogits = logits.map((l) => l.div(Math.max(temperature, 0.01)));
      const probs = this.softmax(scaledLogits);

      // Sample
      const weights = probs.map((p) => p.data);
      tokenId = this.rng.choices(weights);

      if (tokenId === this.tokenizer.BOS) break;

      tokenIds.push(tokenId);
      tokenChars.push(this.tokenizer.tokenToChar(tokenId));

      // Merge attention data
      for (const ad of attentionData) {
        const existing = allAttention.find(
          (a) => a.layer === ad.layer && a.head === ad.head
        );
        if (existing) {
          existing.weights.push(ad.weights[0]);
        } else {
          allAttention.push({ ...ad });
        }
      }
    }

    return {
      text: tokenChars.join(""),
      tokens: tokenIds,
      tokenChars,
      attentionData: allAttention,
    };
  }

  // Generate multiple samples
  generateBatch(count: number, temperature = 0.5): GenerateResult[] {
    const results: GenerateResult[] = [];
    for (let i = 0; i < count; i++) {
      results.push(this.generate(temperature));
    }
    return results;
  }

  // Get model architecture info for visualization
  getArchitectureInfo() {
    const { nEmbd, nHead, nLayer, blockSize, vocabSize } = this.config;
    return {
      layers: [
        { name: "Token Embedding", size: `${vocabSize} x ${nEmbd}`, params: vocabSize * nEmbd },
        { name: "Position Embedding", size: `${blockSize} x ${nEmbd}`, params: blockSize * nEmbd },
        ...Array.from({ length: nLayer }, (_, i) => [
          { name: `Layer ${i} — RMSNorm`, size: `${nEmbd}`, params: 0 },
          { name: `Layer ${i} — Multi-Head Attention`, size: `${nHead} heads, dim=${nEmbd / nHead}`, params: 4 * nEmbd * nEmbd },
          { name: `Layer ${i} — RMSNorm`, size: `${nEmbd}`, params: 0 },
          { name: `Layer ${i} — MLP (FC1 + ReLU + FC2)`, size: `${nEmbd} -> ${4 * nEmbd} -> ${nEmbd}`, params: 2 * 4 * nEmbd * nEmbd },
        ]).flat(),
        { name: "LM Head (Logits)", size: `${nEmbd} -> ${vocabSize}`, params: vocabSize * nEmbd },
      ],
      totalParams: this.params.length,
      config: this.config,
    };
  }
}
