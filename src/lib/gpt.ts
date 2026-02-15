/**
 * GPT model — TypeScript port of Karpathy's pure-Python GPT.
 * Runs entirely in the browser: tokenizer, model, training, inference.
 *
 * Architecture: GPT-2 style with rmsnorm (no layernorm), no biases, ReLU (not GeLU).
 * Config: n_embd=16, n_head=4, n_layer=1, block_size=16, vocab_size=27 (a-z + BOS)
 */

import { Value, vsum } from "./value";
import { NAMES } from "./names-data";

// --- Mersenne Twister PRNG (matches Python's random module exactly) ---
class MersenneTwister {
  private mt = new Uint32Array(624);
  private idx = 625;
  private _gaussNext: number | null = null;

  constructor(seed: number) {
    this.seed(seed);
  }

  seed(n: number): void {
    const u = (a: number, b: number) => Math.imul(a, b) >>> 0;
    const key: number[] = [];
    for (let v = n || 0; v > 0; v = Math.floor(v / 0x100000000)) key.push(v & 0xffffffff);
    if (!key.length) key.push(0);

    this.mt[0] = 19650218;
    for (this.idx = 1; this.idx < 624; ++this.idx) {
      this.mt[this.idx] = (u(1812433253, this.mt[this.idx - 1] ^ (this.mt[this.idx - 1] >>> 30)) + this.idx) >>> 0;
    }

    let i = 1, j = 0;
    for (let k = Math.max(624, key.length); k > 0; --k, ++i, ++j) {
      if (i >= 624) { this.mt[0] = this.mt[623]; i = 1; }
      if (j >= key.length) j = 0;
      this.mt[i] = ((this.mt[i] ^ u(this.mt[i - 1] ^ (this.mt[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0;
    }
    for (let k = 623; k > 0; --k, ++i) {
      if (i >= 624) { this.mt[0] = this.mt[623]; i = 1; }
      this.mt[i] = ((this.mt[i] ^ u(this.mt[i - 1] ^ (this.mt[i - 1] >>> 30), 1566083941)) - i) >>> 0;
    }
    this.mt[0] = 0x80000000;
    this.idx = 624;
    this._gaussNext = null;
  }

  int32(): number {
    if (this.idx >= 624) {
      for (let k = 0; k < 624; ++k) {
        const y = (this.mt[k] & 0x80000000) | (this.mt[(k + 1) % 624] & 0x7fffffff);
        this.mt[k] = (this.mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)) >>> 0;
      }
      this.idx = 0;
    }
    let y = this.mt[this.idx++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  random(): number {
    return ((this.int32() >>> 5) * 67108864.0 + (this.int32() >>> 6)) / 9007199254740992.0;
  }

  gauss(mu = 0, sigma = 1): number {
    let z = this._gaussNext;
    this._gaussNext = null;
    if (z === null) {
      const x2pi = this.random() * 2 * Math.PI;
      const g2rad = Math.sqrt(-2 * Math.log(1 - this.random()));
      z = Math.cos(x2pi) * g2rad;
      this._gaussNext = Math.sin(x2pi) * g2rad;
    }
    return mu + z * sigma;
  }

  shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; --i) {
      const k = 32 - Math.clz32(i + 1);
      let r = this.int32() >>> (32 - k);
      while (r > i) r = this.int32() >>> (32 - k);
      const t = arr[i];
      arr[i] = arr[r];
      arr[r] = t;
    }
  }

  choices(population: number[], weights: number[]): number {
    const cum = new Float64Array(weights.length);
    cum[0] = weights[0];
    for (let i = 1; i < weights.length; i++) cum[i] = cum[i - 1] + weights[i];
    const x = this.random() * cum[cum.length - 1];
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      x < cum[mid] ? hi = mid : lo = mid + 1;
    }
    return population[lo];
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
  const charToId = new Map(uchars.map((ch, i) => [ch, i]));
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;

  return {
    uchars,
    BOS,
    vocabSize,
    encode: (s: string) => [BOS, ...Array.from(s, (ch) => charToId.get(ch)!), BOS],
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
  private rng: MersenneTwister;
  private docs: string[];

  // Adam optimizer state
  private m: Float64Array;
  private v: Float64Array;
  private step: number;
  private learningRate: number;
  private beta1: number;
  private beta2: number;
  private epsAdam: number;

  constructor(seed = 42) {
    this.rng = new MersenneTwister(seed);
    this.docs = [...NAMES];
    this.rng.shuffle(this.docs);
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
    this.m = new Float64Array(this.params.length);
    this.v = new Float64Array(this.params.length);
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
    const ms = vsum(x.map((xi) => xi.mul(xi))).mul(1 / x.length);
    const s = ms.add(1e-5).pow(-0.5);
    return x.map((xi) => xi.mul(s));
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
    const scale = 1 / headDim ** 0.5;
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
          vsum(qH.map((qj, j) => qj.mul(kt[j]))).mul(scale)
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

    const loss = vsum(losses).mul(1 / n);

    // Backward
    loss.backward();

    // Adam update
    const lrT = this.learningRate * (1 - this.step / numSteps);
    const bc1 = 1 - this.beta1 ** (this.step + 1);
    const bc2 = 1 - this.beta2 ** (this.step + 1);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * p.grad;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * p.grad ** 2;
      const mHat = this.m[i] / bc1;
      const vHat = this.v[i] / bc2;
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
      const scaledLogits = logits.map((l) => l.div(temperature));
      const probs = this.softmax(scaledLogits);

      // Sample
      const tokenPool = Array.from({ length: this.config.vocabSize }, (_, i) => i);
      const weights = probs.map((p) => p.data);
      tokenId = this.rng.choices(tokenPool, weights);

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
