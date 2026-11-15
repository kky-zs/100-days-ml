## Progress

### Phase 1: Practical ML core
- [X] Day 1: Training framework
- [ ] Day 2: Autodiff and optimizers

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
