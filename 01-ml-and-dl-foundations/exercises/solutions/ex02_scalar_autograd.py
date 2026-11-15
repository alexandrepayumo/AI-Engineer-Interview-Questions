"""Exercise 2 - A scalar autograd engine (micrograd style).

TASK
    Implement a `Value` class that records a computational graph on the forward
    pass and computes gradients with reverse-mode differentiation.

    Value(data: float, _parents: tuple = (), _op: str = "")
        .data (float), .grad (float, starts at 0.0)
    Operators: +, -, * and / between Values and plain floats (both orders),
        unary minus, and `** k` for a constant int/float k.
    Methods: .tanh(), .relu(), .exp(), .log()
    .backward()
        Seeds self.grad = 1.0 and propagates gradients to every ancestor, using a
        topological order. Gradients must ACCUMULATE (+=) so that a node used twice
        receives the sum of both contributions.

    numeric_grad(f, x: list[float], eps: float = 1e-6) -> list[float]
        Central finite-difference gradient of a scalar function f(list[float]).

HINTS
    - Each operation stores a small closure `_backward` that adds
      (local derivative * out.grad) into the parents' grad.
    - Build the topological order with an iterative or recursive DFS from the
      output; run the closures in reverse order.
    - Subtraction and division can be built from + , * and ** (-1).
    - d tanh(x) = 1 - tanh(x)^2, d relu(x) = 1 if x > 0 else 0,
      d exp(x) = exp(x), d log(x) = 1 / x.

WHAT YOU SHOULD LEARN
    - Backprop is the chain rule applied in reverse topological order, with
      gradient accumulation at fan-out nodes.
    - Why a framework caches forward values (the local derivatives need them).
    - How to gradient-check any differentiable code against finite differences.
"""
from __future__ import annotations

import math
from typing import Callable


class Value:
    def __init__(self, data: float, _parents: tuple = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a  (uses cached forward values)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward() -> None:
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        # Topological order: every node appears after all of its parents.
        order: list[Value] = []
        seen: set[int] = set()

        def visit(v: "Value") -> None:
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._parents:
                visit(p)
            order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):  # outputs first, inputs last
            v._backward()

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other) * (self ** -1)

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


def numeric_grad(f: Callable[[list[float]], float], x: list[float], eps: float = 1e-6) -> list[float]:
    g = []
    for i in range(len(x)):
        hi = list(x)
        lo = list(x)
        hi[i] += eps
        lo[i] -= eps
        g.append((f(hi) - f(lo)) / (2 * eps))
    return g


def _run_tests() -> None:
    # simple expression: f = (a * b + c) ** 2 with a=2, b=-3, c=10 -> (4)^2 = 16
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    f = (a * b + c) ** 2
    f.backward()
    assert abs(f.data - 16.0) < 1e-12
    assert abs(a.grad - 2 * 4 * -3.0) < 1e-12  # 2*(ab+c) * b = -24
    assert abs(b.grad - 2 * 4 * 2.0) < 1e-12   # 16
    assert abs(c.grad - 2 * 4) < 1e-12         # 8

    # fan-out: x used twice, f = x*x + x -> df/dx = 2x + 1
    x = Value(3.0)
    f = x * x + x
    f.backward()
    assert abs(x.grad - 7.0) < 1e-12

    # the classic neuron: o = tanh(x1*w1 + x2*w2 + b)
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    bias = Value(6.8813735870195432)
    o = (x1 * w1 + x2 * w2 + bias).tanh()
    o.backward()
    assert abs(o.data - 0.7071067811865476) < 1e-9
    assert abs(x1.grad - (-1.5)) < 1e-6
    assert abs(w1.grad - 1.0) < 1e-6
    assert abs(x2.grad - 0.5) < 1e-6
    assert abs(w2.grad - 0.0) < 1e-6

    # reflected operators and division with plain floats
    x = Value(4.0)
    f = 1.0 - x / 2.0 + 3.0 * x + (2.0 / x)
    f.backward()
    assert abs(f.data - (1 - 2 + 12 + 0.5)) < 1e-12
    assert abs(x.grad - (-0.5 + 3.0 - 2.0 / 16.0)) < 1e-12

    # gradient check on a composite function with all unary ops
    def build(vals):
        p, q = vals
        v1, v2 = Value(p), Value(q)
        out = (v1 * v2).tanh() + (v1 + 3.0).log() * v2.exp() - (v2 - v1).relu() + v1 / (v2 * v2 + 1.0)
        return v1, v2, out

    pt = [0.7, 0.4]
    v1, v2, out = build(pt)
    out.backward()
    num = numeric_grad(lambda vals: build(vals)[2].data, pt)
    assert abs(v1.grad - num[0]) < 1e-6, (v1.grad, num[0])
    assert abs(v2.grad - num[1]) < 1e-6, (v2.grad, num[1])

    # relu gate: zero gradient for negative input
    x = Value(-2.0)
    y = x.relu() * 5.0
    y.backward()
    assert x.grad == 0.0 and y.data == 0.0

    # train a single neuron by gradient descent: fit y = 2x + 1 with w, b
    w, bb = Value(0.0), Value(0.0)
    data = [(-1.0, -1.0), (0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]
    for _ in range(200):
        w.grad = 0.0
        bb.grad = 0.0
        loss = Value(0.0)
        for xi, yi in data:
            loss = loss + (w * xi + bb - yi) ** 2
        loss = loss * (1.0 / len(data))
        loss.backward()
        w.data -= 0.1 * w.grad
        bb.data -= 0.1 * bb.grad
    assert abs(w.data - 2.0) < 1e-3 and abs(bb.data - 1.0) < 1e-3

    # numeric_grad itself
    g = numeric_grad(lambda v: v[0] ** 2 + 3 * v[1], [2.0, 5.0])
    assert abs(g[0] - 4.0) < 1e-5 and abs(g[1] - 3.0) < 1e-5


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
