"""Job rotation at the breaks (online version of Sec. 4.2 / Job-Rotation-TL).

Terminology: a LOOP is the group of tasks the MILP assigns to one operator slot (fixed during the
shift); an OPERATOR is a person. Rotation decides which operator works on which loop in each block.

Sec. 4.2 plans all R periods at the start of the shift from traffic-light scores. Here the decision
is taken at each break with what has been MEASURED so far. Objective, as in job_rotation.py:
minimise E_max - E_min of the cumulative exposure after the next block (tie-break: smallest E_max),
where each operator's cumulative exposure can optionally be adjusted by the Borg rating:

    E~_o = E_o + borg_weight * (Borg_o / 10) * mean forecast exposure of one block

By default borg_weight = 0: Borg enters the rotation only through the Borg rule (constraint).

Constraints
  * eligibility (as in Job-Rotation-TL): an operator can work on a loop only if qualified and
    medically cleared for every workstation of the loop
  * Borg rule: an operator who reports Borg >= borg_limit is not sent to the heaviest loop
If no assignment satisfies them, the Borg rule is dropped first, then eligibility, rather than
stopping the line; the relaxation is reported.

With at most 7-8 operators all assignments are enumerated (7! = 5040), so the result is exact.
"""
from itertools import permutations


def rotate(operators, loops, cumulative, block_load, borg=None, borg_limit=7, borg_weight=0.0,
           eligible=None):
    """operators: operators present; loops: loop ids; cumulative: {operator: exposure so far};
    block_load: {loop: forecast exposure of the next block}; borg: {operator: last Borg};
    eligible(operator, loop) -> bool. Returns ({loop: operator}, relaxed) with relaxed in
    {None, "borg_rule", "eligibility"}."""
    borg = borg or {}
    mean_block = sum(block_load.values()) / len(block_load)
    eff = {o: cumulative[o] + borg_weight * borg.get(o, 0) / 10 * mean_block for o in operators}
    heaviest = max(loops, key=lambda s: block_load[s])
    tired = {o for o in operators if borg.get(o, 0) >= borg_limit}

    def search(use_borg_rule, use_eligibility):
        best, best_key = None, None
        for perm in permutations(operators):
            pairs = list(zip(loops, perm))
            if use_borg_rule and any(o in tired and s == heaviest for s, o in pairs):
                continue
            if use_eligibility and eligible and not all(eligible(o, s) for s, o in pairs):
                continue
            after = [eff[o] + block_load[s] for s, o in pairs]
            key = (max(after) - min(after), max(after))
            if best_key is None or key < best_key:
                best, best_key = dict(pairs), key
        return best

    best = search(True, True)
    if best is not None:
        return best, None
    best = search(False, True)
    if best is not None:
        return best, "borg_rule"
    return search(False, False), "eligibility"
