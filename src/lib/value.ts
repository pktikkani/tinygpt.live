/**
 * Autograd Value class — TypeScript port of Karpathy's scalar autograd engine.
 * Optimized to match microgpt.js: direct child fields + generation counter for topo sort.
 */

let _gen = 0; // global generation counter for topological sorting

export class Value {
  data: number;
  grad: number;
  _c0: Value | undefined;  // first child
  _c1: Value | undefined;  // second child
  _lg0: number;            // local grad w.r.t. first child
  _lg1: number;            // local grad w.r.t. second child
  _nch: number;            // number of children (0, 1, or 2)
  _gen: number;            // generation marker for topo sort

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.grad = 0;
    this._c0 = children[0];
    this._c1 = children[1];
    this._lg0 = localGrads[0] ?? 0;
    this._lg1 = localGrads[1] ?? 0;
    this._nch = children.length;
    this._gen = 0;
  }

  add(other: Value | number): Value {
    if (other instanceof Value) return new Value(this.data + other.data, [this, other], [1, 1]);
    return new Value(this.data + other, [this], [1]);
  }

  mul(other: Value | number): Value {
    if (other instanceof Value) return new Value(this.data * other.data, [this, other], [other.data, this.data]);
    return new Value(this.data * other, [this], [other]);
  }

  pow(exp: number): Value {
    return new Value(this.data ** exp, [this], [exp * this.data ** (exp - 1)]);
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }

  relu(): Value {
    return new Value(Math.max(0, this.data), [this], [+(this.data > 0)]);
  }

  neg(): Value {
    return new Value(-this.data, [this], [-1]);
  }

  sub(other: Value | number): Value {
    return this.add(other instanceof Value ? other.neg() : -other);
  }

  div(other: Value | number): Value {
    return this.mul(other instanceof Value ? other.pow(-1) : 1 / other);
  }

  backward(): void {
    const gen = ++_gen;
    const topo: Value[] = [];

    function buildTopo(v: Value) {
      if (v._gen === gen) return;
      v._gen = gen;
      if (v._nch >= 1) buildTopo(v._c0!);
      if (v._nch === 2) buildTopo(v._c1!);
      topo.push(v);
    }

    buildTopo(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; --i) {
      const v = topo[i];
      const g = v.grad;
      if (v._nch >= 1) v._c0!.grad += v._lg0 * g;
      if (v._nch === 2) v._c1!.grad += v._lg1 * g;
    }
  }
}

// Helper: scalar multiply Value by number from left side
export function smul(n: number, v: Value): Value {
  return v.mul(n);
}

// Helper: sum an array of Values
export function vsum(vals: Value[]): Value {
  return vals.reduce((a, b) => a.add(b));
}
