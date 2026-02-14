/**
 * Autograd Value class — TypeScript port of Karpathy's scalar autograd engine.
 * Tracks computation graphs and computes gradients via backpropagation.
 */

export class Value {
  data: number;
  grad: number;
  private _children: Value[];
  private _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(exp: number): Value {
    return new Value(
      this.data ** exp,
      [this],
      [exp * this.data ** (exp - 1)]
    );
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }

  relu(): Value {
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1 : 0]
    );
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) {
          buildTopo(child);
        }
        topo.push(v);
      }
    };

    buildTopo(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) {
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}

// Helper: scalar multiply Value by number from left side
export function smul(n: number, v: Value): Value {
  return v.mul(n);
}

// Helper: sum an array of Values
export function vsum(vals: Value[]): Value {
  let result = vals[0];
  for (let i = 1; i < vals.length; i++) {
    result = result.add(vals[i]);
  }
  return result;
}
