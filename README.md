## Progress

### Phase 1: Practical ML core
- [X] Day 1: Training framework
- [X] Day 2: Autodiff and optimizers

## Daily Takeaways

<details>
<summary><strong>Day 1 — Training Framework</strong></summary>

### 1. Five steps of a training iteration, and what breaks if one is missing

*zero_grad() → forward pass → compute loss → backward() → optimizer.step()*

**What breaks if one is missing?**

- Without `zero_grad()`, gradients accumulate across batches and updates become wrong or diverge.
- Without `backward()`, there are no gradients and nothing learns.
- Without `step()`, gradients exist but the weights never change.
- The forward pass and loss are what produce the graph that `backward()` walks.

### 2. `model.eval()` vs `torch.no_grad()`

- `model.eval()` changes layer behaviour:
  - Dropout is turned off.
  - BatchNorm uses running statistics instead of batch statistics.
- `torch.no_grad()` stops PyTorch from building the computation graph, which saves memory and time.

They are independent, and for validation we want both.

- Missing `eval()` gives noisy, pessimistic metrics.
- Missing `no_grad()` only wastes memory.

### 3. What belongs in a resumable checkpoint, and why?

A resumable checkpoint should contain:

- Model weights
- Optimizer state (Adam’s moving averages)
- Scheduler state
- The epoch counter
- The training history
- The best validation loss
- The early-stopping counter
- The configuration

Weights alone would restart Adam’s statistics and the LR schedule from scratch, so "resume" would quietly behave like a fresh start with warm weights.

### 4. What did overfit-one-batch tell you?

That the whole pipeline can learn:

- Data and labels line up.
- The loss function is right.
- Gradients flow.
- The optimizer updates weights.
- The model has enough capacity.

**Results:**
- Loss ≈ `0.0003`
- Accuracy = `1.0`

It doesn’t prove the model generalizes, only that nothing is fundamentally broken.

### 5. Which of the four self-checks failed first, and what was the cause?

In my run, none failed. All four passed.
</details>

<details>
<summary><strong>Day 2 — Autodiff and Optimizers</strong></summary>

### 1. Why does every `_backward` use `+=`?

A value can be used in several places, and its gradient is the sum of the contributions from each use (multivariable chain rule).

### 2. Why reverse mode and not forward mode?

A network has millions of inputs (weights) and one scalar output (the loss).

- **Reverse mode:** Gets the gradient with respect to all inputs in about one backward pass (cost roughly a small multiple of the forward pass).
- **Forward mode:** Needs one pass per input.

The price of reverse mode is memory: it has to store the graph and intermediate values.

### 3. What does a numerical gradient check do, and why only for testing?

It estimates each derivative with finite differences:

`(f(x+h) - f(x-h)) / (2h)`

And compares that with the analytic gradient.

- A large relative error means a bug in a backward rule.
- It needs two forward passes per parameter, so it’s far too slow for training.
- It’s also numerically delicate (choice of `h`, trouble near kinks like ReLU at 0).

### 4. What problem does Adam’s bias correction solve?

`m` and `v` start at zero, so in the first steps they’re biased toward zero.

- For `v`, that makes the step `m/√v` too large or too erratic.
- For `m`, it shrinks early updates.

Dividing by `1 − βᵗ` removes that bias and the correction fades out as `t` grows.

### 5. Adam with L2 vs AdamW

**Adam + L2**

Adds `wd·p` to the gradient. That term then goes through the adaptive scaling, so the effective decay differs per parameter (weights with large gradient history get less decay).

**AdamW**

Shrinks the weights directly:

```python
p ← p(1 − lr·wd)
```

Separately from the gradient step. The decay is uniform and the regularization behaves as intended.

It’s the standard choice for transformers.

### 6. Which optimizer did best, and which would you pick for a real model?

**Rosenbrock optimization results:**

| Optimizer | Final function value |
|---|---:|
| Adam | 1.7e-12 |
| Momentum | 1.4e-8 |
| SGD | 4.6e-5 |
| RMSprop | 8.6e-3 |

Adam ended best, then momentum, SGD and RMSprop.

RMSprop was hampered by its constant learning rate, which makes it hover near the minimum, and SGD by an early overshoot.

</details>
