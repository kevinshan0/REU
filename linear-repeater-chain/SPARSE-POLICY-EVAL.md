# Sparse policy evaluation (`method="direct"`)

How `src/policy.py` uses SciPy's sparse linear algebra to speed up policy
iteration, why it helps, and when to use it.

## The bottleneck: policy evaluation

Policy iteration alternates two steps until the policy stops changing:

1. **Policy evaluation** — given a fixed policy $\pi$, compute $V^\pi(s)$ for
   every state.
2. **Policy improvement** — at every state, re-pick the action that maximizes
   the Q-value under the current $V$.

Improvement is one pass over the state space, so its cost is fixed. Evaluation
is where the time goes, and how it is done is the choice this document is about.

Our MDP is undiscounted ($\gamma = 1$) with reward $-1$ per time slot, so
$V(s) = -\,\mathbb{E}[\text{time slots to end-to-end entanglement from } s]$,
and the expected delivery time from the empty state is `-(state_info[0]['value'] + 1)`.
Terminal (end-to-end-linked) states are absorbing with $V = 0$ by convention.

### The default: iterative evaluation

`_evaluate_policy_iterative` (policy.py:548) does what the reference
implementation of Iñesta et al. (arXiv:2207.06533) does — successive
approximation. Sweep the Bellman backup

$$V_{k+1}(s) \;=\; \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a)\,\bigl(-1 + V_k(s')\bigr)$$

over every state, over and over, until the largest per-state change falls below
a tolerance.

Each sweep propagates information exactly **one time step of the horizon**. So
the number of sweeps needed scales with how long the chain takes to deliver —
that is, with how slowly it mixes. At $p = 0.9$ an entangled link appears almost
immediately and a handful of sweeps suffice; at $p = 0.25, p_s = 0.35$ the
expected delivery time is ~74 time slots, and the truncated Bellman sum needs
hundreds of sweeps before enough probability mass has been accounted for to
satisfy an absolute tolerance. **The physics of the slow regime is exactly the
regime the iterative method handles worst.**

### The fix: solve the linear system instead

For a *fixed* policy, the Bellman equation is not a fixed-point to be iterated
toward — it is a linear system that can be solved outright. Writing $P_\pi$ for
the policy-induced transition matrix over non-terminal states, the equation
above is

$$V \;=\; -\mathbf{1} + P_\pi V \qquad\Longleftrightarrow\qquad (I - P_\pi)\,V \;=\; -\mathbf{1}$$

which is $|S|$ linear equations in $|S|$ unknowns. Terminal states contribute
their $-1$ reward but no column (their $V$ is pinned at 0), so they are dropped
from the unknown vector entirely.

`_evaluate_policy_direct` (policy.py:592) builds and solves this system.
**A direct solve's cost depends on the sparsity structure of the transition
graph, not on how slowly the chain mixes** — the horizon factor disappears
completely.

## How the SciPy part works

```python
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
```

Three details make the construction work:

**1. Accumulate coefficients in a dict, not a matrix.** The state space is
discovered lazily (`agent.observe()` registers states as the walk finds them),
so the matrix dimensions are not known until the walk finishes. We accumulate

```python
coeffs[(row_idx, col_idx)] -= action_prob * P
```

into a plain dict keyed by `(from_state, to_state)`, walking `agent.state_list`
exactly the way the iterative version does — same discovery order, same
`cached_step` transition cache. Only the bookkeeping differs.

**2. COO for assembly, CSC for solving.** The dict becomes three flat lists
(`rows`, `cols`, `data`) and then a `coo_matrix`. Coordinate format is the right
choice for *building* a sparse matrix from unordered triplets — it is just three
arrays, with no index structure to maintain per insertion. It is the wrong
format for solving, so `.tocsc()` converts to compressed-sparse-column, which is
what SuperLU wants.

One convenience worth knowing: **`coo_matrix` sums duplicate entries on
conversion.** That is what makes the $I$ term a one-liner —

```python
for agent_idx in non_terminal_indices:
    rows.append(row_of[agent_idx]); cols.append(row_of[agent_idx]); data.append(1.0)
```

A state with a self-loop already contributed $-P$ at its diagonal position; the
appended $+1$ is summed with it, correctly giving $1 - P$. No special-casing.

**3. One `spsolve`.** `spsolve(A, b)` with `b = -np.ones(size)` runs a sparse LU
factorization (SuperLU) and a triangular solve. The result is written straight
back into the agent's `value` field.

### The matrices really are sparse

Measured `nnz` of $(I - P_\pi)$ at the final evaluation pass:

| Configuration | size | nnz | density | nnz/row | dense storage |
|---|---|---|---|---|---|
| n=4, t_cut=3, no discard | 215 | 1,773 | 3.8% | 8.2 | 0.4 MB |
| n=5, t_cut=2, no discard | 536 | 8,225 | 2.9% | 15.3 | 2.3 MB |
| n=4, t_cut=4, discard cap 1 | 3,008 | 18,320 | 0.20% | 6.1 | 72 MB |
| n=5, t_cut=2, discard cap 1 | 3,336 | 45,713 | 0.41% | 13.7 | 89 MB |

Nonzeros per row stays roughly constant (~6–15) as the state space grows,
because a state's successor count is set by the local branching of the dynamics
— which links generate, which swaps succeed — not by how many states exist.
Density therefore *falls* as the problem grows, which is precisely the regime
sparse LU is built for. A dense solve would be quadratic in memory and cubic in
time for no benefit.

## Measured speedup

Full `policy_iteration` wall-clock, `tolerance=1e-5`, same machine, fresh
process each run (so neither method benefits from the other's `cached_step`
cache):

| n | p | p_s | t_cut | discard | states | iterative | direct | speedup |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.9 | 0.5 | 3 | — | 231 | 0.10 s | 0.06 s | 1.6× |
| 4 | 0.3 | 0.5 | 3 | — | 231 | 0.29 s | 0.06 s | 5.2× |
| 5 | 0.9 | 0.5 | 2 | — | 563 | 1.32 s | 0.45 s | 2.9× |
| 4 | 0.25 | 0.35 | 3 | — | 231 | 0.70 s | 0.06 s | 12.6× |
| 4 | 0.9 | 0.5 | 4 | cap 1 | 3,033 | 12.56 s | 0.88 s | 14.2× |
| 5 | 0.9 | 0.5 | 2 | cap 1 | 3,399 | 97.36 s | 4.59 s | 21.2× |
| 4 | 0.25 | 0.35 | 3 | cap 1 | 1,397 | 103.54 s | 0.46 s | **224.6×** |

The sweep counts explain the whole table. Direct evaluation performs exactly
**one solve per outer iteration** — 4–5 total, since the outer loop itself
converges in a handful of iterations. Iterative evaluation needed 58 sweeps in
the first row, 863 in the fourth, and **8,944** in the last. The two axes that
blow up the sweep count are:

- **Slow mixing (low `p`, low `p_s`).** Row 4 has the same 231 states as row 1
  but a ~74-slot expected delivery time instead of ~4.8, and needs 15× the
  sweeps.
- **`--discard`, which inflates the state and action space.** Every extra state
  is another state each sweep must revisit; the sparse solve absorbs them into
  one factorization instead.

Both compounding (last row) is where the method goes from *nice* to *necessary*.

## Accuracy

Direct evaluation is also **more accurate**, not a speed-for-precision trade.
Each iterative evaluation stops as soon as it is within `tolerance`, so it
returns a value still short of the true $V^\pi$ by roughly that much; the
direct solve returns the exact solution up to LU round-off. On the
n=4, p=0.25, p_s=0.35, t_cut=3 case the two converge to 73.63083 and 73.63034
expected time slots — the iterative figure is the one carrying the tolerance
error.

One consequence worth knowing about, documented in `policy_iteration`'s
docstring: because exact evaluation makes Q-values at some states numerically
*tied*, the improvement step can cycle forever between two equally-good actions
without the policy ever registering as "exactly stable". That is why the outer
loop stops on **either** an exactly stable policy for two consecutive iterations
**or** a value change below `tolerance`. The value-based check exists
specifically to catch that case, and it is what makes `method="direct"`
terminate.

## Usage

```bash
# CLI
python src/policy.py --n 5 --p 0.9 --p_s 0.5 --cutoff 2 --method direct
python src/policy.py --n 4 --p 0.25 --p_s 0.35 --cutoff 3 --discard 1 --method direct
```

```python
# API
from policy import policy_iteration
v0_evol, state_info, exe_time = policy_iteration(
    n=5, p=0.9, p_s=0.5, cutoff=2, tolerance=1e-5,
    allow_discard=1, method="direct")
expected_delivery_time = -(state_info[0]["value"] + 1)
```

`method` defaults to `"iterative"` to stay faithful to the paper's algorithm.
Pass `method="direct"` for anything at low `p`/`p_s`, with `--discard`, or at
larger `n`/`t_cut`. Note `v0_evol` changes shape between the two: iterative
records one entry per sweep (a convergence trace), direct records a single exact
value per outer iteration, since there is no sweep-by-sweep convergence to
trace.

`--policy swap-asap` evaluates one fixed policy and does not take `--method`;
it uses iterative evaluation.

## The same trick elsewhere in the codebase

Once policy evaluation is a sparse solve, the same $(I - P)$ assembly shows up
wherever a fixed policy has to be scored exactly:

| Location | What it solves |
|---|---|
| `policy.py:_evaluate_policy_direct` | $(I - P_\pi)V = -\mathbf{1}$ — the evaluation step described above |
| `policy.py:_relaxed_optimal_cost_table` | Exact optimum of the relaxed (no-decoherence) chain, used as BRTDP's admissible lower bound. Self-contained, small enough that iterative sweeps would be pure overhead |
| `policy.py:_solve_envelope` | BRTDP's envelope fixed point, with out-of-envelope successors pinned at their current bound. Removes the same horizon factor from BRTDP that `method="direct"` removes from policy iteration |
| `policy.py:bounded_rtdp_policy` | One final `_evaluate_policy_direct` pass to give the frozen hybrid policy exact values rather than raw trial bounds |
| `distill.py:_visitation_counts` | Expected visitation counts — the **transposed** system $(I - P)^\top x = e_0$, solved against an indicator at the initial state instead of an all-ones vector |
| `local_policy.py`, `po_policy.py`, `info_delay_solver.py` | Same two patterns for the local-policy, POMDP, and information-delay models |

The transpose case is the one worth flagging: value functions and visitation
counts are the same matrix read in opposite directions. $V$ answers "what does
the future cost from here?" and satisfies $(I-P)V = -\mathbf{1}$; visitation
counts answer "how often do we arrive here?" and satisfy
$x^\top(I - P) = e_0^\top$. One assembly, two questions.
