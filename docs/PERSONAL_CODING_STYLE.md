# Personal coding style (Python)

My preferred Python style for this project. Follow it when writing **new** code here.

**One rule above all: write for the human who reads this next — not for the type checker.**

**Who that reader is:** a scientist who has worked through a solid *basic* Python-and-type-hints
tutorial. Assume they are comfortable with functions, loops, lists and dicts, simple classes, list
comprehensions, and plain hints like `x: int` and `list[float]` — and *not* with exotic typing,
nested generics, library-specific idioms, or clever one-liners. **The test for any line:** could that
reader read it once and understand it, without looking anything up? If not, simplify until they can.
The same bar covers a programmer fluent in another language (C, C++, R) but new to Python: each line
should be graspable at a glance. So avoid Python-only special cases — e.g. `None` used to mean "insert
an axis" — that read wrongly unless you already know the trick (`None` normally means "nothing"; numpy
even ships `np.newaxis` because `None` is a bad spelling for it).

Prefer plain, simple Python with clear names and inline comments. Reach for a type annotation only
when **(a) a CI type checker actually consumes it**, or **(b) the type is simple *and* adds meaning**.
Otherwise a good name plus a full-sentence comment communicates better than a type ever will: Python
annotations are unenforced hints, not C++ declarations, so an unchecked annotation is just
documentation wearing a type's clothes.

---

## 1. Types must earn their keep

- **Keep** simple, meaningful, checker-verified types: `list[float]`, a function's public signature,
  a named return value.
- **Drop** types that don't earn it:
  - *Redundant* — `num_particles: int = 1` (the `1` already says it is an int). Write `num_particles=1`.
  - *Unenforced* — `Float[Tensor, "*batch atoms 3"]` with no runtime checker is a cryptic comment, not
    a guarantee. Put the shape in a plain sentence instead.
  - *Structure, not meaning* — `list[list[float]]` tells you the nesting, not what it *is*.
- **Readability threshold.** `list[float]` (one familiar level) is tutorial-level and fine. The
  moment a type nests (`list[list[float]]`), leans on an opaque domain type (`list[Tensor]`), or
  compounds into a positional tuple (`tuple[Tensor, list[Tensor], ...]`) — none of which a basic
  tutorial covers — stop: a name, a comment, or a small `namedtuple` reads better than the type.

## 2. Prefer several simple lines over one clever line

One complicated expression forces the reader to unpack it in their head. Split it, name the pieces,
and comment the intent — **one operation per line.** A chain like `x.max(dim=0)[0].relu()` makes the
reader parse three transforms right-to-left; give each its own named line instead.

```python
# Avoid -- one dense line, two constructions, nothing named:
return torch.zeros_like(state), torch.zeros(state.shape[0], device=state.device)

# Prefer -- one idea per line, named, commented, and dtype-correct:
direction = torch.zeros_like(state)        # no coordinate move this step
loss = state.new_zeros(state.shape[0])     # zero loss, one value per batch member
return direction, loss
```

```python
# Avoid -- nested calls, no explanation:
return torch.zeros_like(torch.as_tensor(context.t_effective))

# Prefer -- name it and say what it is:
# t_effective is the per-member noise level, shape [batch]; we want a matching all-zero weight.
return torch.zeros_like(context.t_effective)
```

(torch note: `x.new_zeros(...)` and `torch.zeros_like(x)` match **both** dtype and device;
`torch.zeros(..., device=x.device)` silently drops the dtype.)

```python
# Avoid -- three operations chained; the reader unpacks it right-to-left to see what happens:
overlap = (min_dist - distances).max(dim=0)[0].relu()

# Prefer -- one operation per line, each nameable and readable coming from any language:
worst = (min_dist - distances).max(dim=0).values   # worst overlap across the ensemble, per pair
overlap = worst.clamp(min=0)                        # keep only genuine clashes (a hinge)
```

## 3. Names and sentences carry meaning; types carry structure

When what a thing *is*, or *why it exists*, is not obvious from the name, write a full-sentence
comment. Do not lean on the type to explain it — a type describes the container's shape, not the
purpose. Say why and what it is connected to first, then how.

## 4. Keep the logic path clear; move plumbing aside

Do not drop bookkeeping (typed accumulators, logging) into the middle of the algorithm the reader is
trying to follow. Let the main loop read as the science; keep diagnostics subordinate — collect them
in a plainly named list, or let a helper *report* what it did, rather than declaring
`optimization_losses: list[list[float]] = []` in the doorway of the loop.

## 5. Name what travels together

Prefer a small `namedtuple`/record (or at least a named unpack plus a comment) over a bare positional
tuple return, so a caller reads `final_coords, trajectory, losses = ...` instead of decoding
`tuple[Tensor, list[Tensor], list[float | None]]` and then hunting the call site to learn what each
slot means.

## 6. Match the codebase, but add the human layer

Follow the house conventions where they do not hurt the reader. Where the existing style is dense
(heavy annotations, clever one-liners), do **not** propagate that into new code — add the name and
the sentence on top. Consistency matters, but it is subordinate to readability.

---

*This is guidance for how to write **new** code here. It is not a mandate to rewrite existing files.*
