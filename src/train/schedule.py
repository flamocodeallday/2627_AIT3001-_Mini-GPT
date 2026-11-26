"""Learning-rate schedule: linear warmup, then cosine decay to min_lr. Exercise for Todd (bài 9)."""

from __future__ import annotations

import math  # noqa: F401  (you will need it)


def get_lr(step: int, *, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Learning rate to use at optimizer step `step` (0-indexed).

    * Warmup: for step < warmup_steps, grow linearly so that step 0 is small but non-zero
      (use (step + 1) / warmup_steps) and the last warmup step reaches max_lr.
    * Cosine: for warmup_steps <= step <= total_steps, follow half a cosine from max_lr down to
      min_lr, reaching min_lr exactly at step == total_steps.
    * After total_steps: stay at min_lr.
    Decision C: total_steps is the budget of THIS run, so every model ends its decay at its own end.
    """

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)
