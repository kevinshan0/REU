"""Physical (fidelity-aware) evaluation of the optimal qubit-age cutoff
policy found by policy.py.

policy_iteration optimizes a pure discrete MDP: swaps and cutoffs are binary
success/failure events with no notion of fidelity. To measure how that
policy actually performs on real quantum states under depolarizing noise, we
hand its state->action table to simulate.jl, a QuantumSavory.jl simulator
that runs the exact same policy tick-by-tick against simulated registers and
reports the resulting average end-to-end Bell-pair fidelity.

The pickled state_info (see policy.py's _export_state_info) stores
action_space in the paper's node-index convention, which is not directly
usable to drive environment.py's qubit-index swap semantics. Rather than
inverting that conversion, this module regenerates the raw qubit-index
action list for every state via State.get_actions() -- a pure, deterministic
function of (qubits, Configuration) that reproduces the exact ordering
'policy' was recorded against, whether the policy was just computed or
loaded from an existing pickle.

Pass --local-policy to simulate a strictly local, per-node policy distilled
from the global one instead (see distill.py): each node then decides its own
swap/discard actions from only its own qubits' state, rather than the full
chain state. Add --local-policy-temperature to distill (and simulate)
probabilistically instead of the default greedy majority vote.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess

from distill import build_local_policy_json
from environment import Configuration, State
from policy import (
    _policyiter_filename,
    check_policyiter_data,
    key_to_qubits,
    load_policyiter_data,
    policy_iteration,
)

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
SIMULATE_JL = SRC_DIR / "simulate.jl"

# policy.py's save/load helpers (check_policyiter_data, load_policyiter_data,
# policy_iteration's savedata path) use paths relative to 'data_policyiter/',
# which they resolve against the current process's CWD rather than this
# module's location. Anchoring CWD here means this module behaves the same
# regardless of where the caller invoked it from.
os.chdir(SRC_DIR)


def _export_action(raw_action, allow_discard):
    """Serialize one raw (qubit-index) action for the JSON handoff."""
    if not allow_discard:
        return list(raw_action)
    swap_nodes, discard_qubits = raw_action
    return {"swap": list(swap_nodes), "discard": list(discard_qubits)}


def build_policy_json(n, p, p_s, cutoff, tolerance, allow_discard, out_path):
    """Regenerate the raw qubit-index action space for every state in the
    saved policy data and write {qubits, policy, actions} triples as JSON
    for simulate.jl to consume."""
    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    _, state_info, _ = load_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard)

    states = []
    for entry in state_info:
        qubits = key_to_qubits(entry["qubits"])
        raw_actions = State(qubits, parameters).get_actions()
        states.append({
            "qubits": entry["qubits"],
            "policy": entry["policy"],
            "actions": [_export_action(a, allow_discard) for a in raw_actions],
        })

    with open(out_path, "w") as handle:
        json.dump({"n": n, "states": states}, handle)


def simulate(n, p, p_s, cutoff, tolerance=1e-5, allow_discard=False,
             episodes=1000, F_new=0.97, tau=10.0, quiet=False, local_policy=False,
             local_policy_temperature=None, local_policy_prob_floor=1e-3,
             local_policy_visitation_weighted=True):
    """Ensure the optimal policy exists (running policy iteration if not),
    then run `episodes` physical simulation episodes of it via simulate.jl
    and return {'avg_fidelity', 'episodes', 'avg_delivery_time'}. If
    local_policy is True, the global-knowledge policy is first distilled
    into a strictly local per-node policy (see distill.py), and that local
    policy is simulated instead -- greedily (deterministic) by default, or
    probabilistically if local_policy_temperature is given (in which case
    local_policy_visitation_weighted also applies -- see distill.py's
    distill_local_policy_soft)."""
    if not check_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard):
        policy_iteration(n, p, p_s, cutoff, tolerance=tolerance, progress=not quiet,
                          savedata=True, allow_discard=allow_discard)

    stem = _policyiter_filename(n, p, p_s, cutoff, tolerance, allow_discard)

    if local_policy:
        if local_policy_temperature is None:
            suffix = "_local"
        else:
            weighting_suffix = "" if local_policy_visitation_weighted else "_unweighted"
            suffix = "_local_soft_tau%.3f%s" % (local_policy_temperature, weighting_suffix)
        policy_json_path = Path(stem + suffix + ".json").resolve()
        policy_json_path.parent.mkdir(parents=True, exist_ok=True)
        build_local_policy_json(n, p, p_s, cutoff, tolerance, allow_discard, policy_json_path,
                                 temperature=local_policy_temperature,
                                 prob_floor=local_policy_prob_floor,
                                 visitation_weighted=local_policy_visitation_weighted)
        result_path = Path(stem + suffix + "_simresult.json").resolve()
    else:
        policy_json_path = Path(stem + ".json").resolve()
        policy_json_path.parent.mkdir(parents=True, exist_ok=True)
        build_policy_json(n, p, p_s, cutoff, tolerance, allow_discard, policy_json_path)
        result_path = Path(stem + "_simresult.json").resolve()

    cmd = [
        "julia", "--project=" + str(REPO_ROOT), str(SIMULATE_JL),
        str(policy_json_path), str(result_path),
        "--n", str(n),
        "--p", str(p),
        "--p_s", str(p_s),
        "--cutoff", str(cutoff),
        "--episodes", str(episodes),
        "--F_new", str(F_new),
        "--tau", str(tau),
    ]
    if allow_discard is not False:
        cmd += ["--discard", str(int(allow_discard) if allow_discard is not True else -1)]
    if local_policy:
        cmd += ["--local", "1"]

    # simulate.jl prints its own summary line too (useful when run standalone
    # for debugging), which would otherwise duplicate the one this module
    # prints below -- suppress it here, but leave stderr passing through so
    # a real Julia error is still visible.
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    with open(result_path) as handle:
        return json.load(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the optimal qubit-age cutoff policy through the "
                     "QuantumSavory physical simulator and report average "
                     "end-to-end fidelity.")
    parser.add_argument("--n", type=int, default=3, help="Number of nodes.")
    parser.add_argument("--p", type=float, default=0.8,
                        help="Elementary link generation success probability.")
    parser.add_argument("--p_s", type=float, default=0.8,
                        help="Entanglement swap success probability.")
    parser.add_argument("--cutoff", type=int, default=5, help="Qubit-age cutoff.")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Policy iteration value tolerance.")
    parser.add_argument("--discard", nargs="?", const=True, default=False, type=int,
                        help="Allow the agent to discard occupied qubits each time "
                             "slot (see policy.py). Must match a previously computed "
                             "policy's setting.")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of simulated delivery episodes to average over.")
    parser.add_argument("--F_new", type=float, default=0.97,
                        help="Fidelity of a freshly generated elementary Bell pair.")
    parser.add_argument("--tau", type=float, default=10.0,
                        help="Mean depolarization interval, in the same tick units "
                             "as --cutoff.")
    parser.add_argument("--local-policy", action="store_true",
                        help="Distill the global-knowledge policy into a strictly local, "
                             "per-node policy (see distill.py) and simulate that instead "
                             "of the global policy.")
    parser.add_argument("--local-policy-temperature", type=float, default=None,
                        help="Distill --local-policy probabilistically, temperature-scaling "
                             "the (visitation-weighted, by default) weighted-majority vote "
                             "-- see distill.py's distill_local_policy_soft -- instead of "
                             "the default greedy majority vote. Implies --local-policy.")
    parser.add_argument("--local-policy-prob-floor", type=float, default=1e-3,
                        help="Only with --local-policy-temperature: prune candidate local "
                             "actions below this probability before simulating them.")
    parser.add_argument("--local-policy-uniform-states", action="store_true",
                        help="Only with --local-policy-temperature: weight every matching "
                             "global state's vote equally instead of by its expected "
                             "visitation count under the optimal policy -- see distill.py's "
                             "distill_local_policy_soft. Independent of temperature.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args()

    # have to set τ_qs given to QS's depolarization noise channel to 2 * τ_paper becuase
    # Paper (per qubit): survival probability = e^(−Δt / (2·τ_paper))
    # QuantumSavory (per qubit): survival probability = e^(−Δt / τ_QS)
    result = simulate(args.n, args.p, args.p_s, args.cutoff, tolerance=args.tol,
                       allow_discard=args.discard, episodes=args.episodes,
                       F_new=args.F_new, tau=2 * args.tau, quiet=args.quiet,
                       local_policy=args.local_policy or args.local_policy_temperature is not None,
                       local_policy_temperature=args.local_policy_temperature,
                       local_policy_prob_floor=args.local_policy_prob_floor,
                       local_policy_visitation_weighted=not args.local_policy_uniform_states)

    print("Episodes: %d, Avg. delivery time: %.4f, Avg. fidelity: %.6f" % (
        result["episodes"], result["avg_delivery_time"], result["avg_fidelity"]))
