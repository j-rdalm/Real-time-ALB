"""Ergo-ALBP MILP of Sec. 4.4 (assembly_line_new_model.mod, OBJ-1), rewritten for PuLP + HiGHS.

    min  w_t * CT + w_e * EL                         (lexicographic tie-break: then min CT + EL)
    s.t. each task on exactly one operator
         CT >= time load of every operator, EL >= ergonomic load of every operator
         precedence: sum_s s*x[pred,s] <= sum_s s*x[succ,s]
         workstation contiguity in operator order (y, firstOp, lastOp)

The number of operators m is a parameter, so the same model is solved for 4, 5, 6 or 7 operators.
"""
import time
import pulp


def solve_balance(t, e, ws, precedence, m, w_time=0.5, w_eri=0.5, time_limit=120,
                  lexicographic=True, msg=False, warm_start=None):
    """t, e: dict task -> normalised time / ergonomic load. ws: dict task -> workstation 1..p.
    Returns dict with assignment {task: operator}, CT, EL, objective, status, seconds."""
    tasks = sorted(t)
    S = range(1, m + 1)
    W = sorted(set(ws.values()))
    p_ws = max(W)

    prob = pulp.LpProblem("ErgoALBP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (tasks, S), cat="Binary")
    y = pulp.LpVariable.dicts("y", (S, W), cat="Binary")
    last = pulp.LpVariable.dicts("lastOp", W, lowBound=0, upBound=m, cat="Integer")
    first = pulp.LpVariable.dicts("firstOp", W, lowBound=1, upBound=m, cat="Integer")
    CT = pulp.LpVariable("CT", lowBound=0)
    EL = pulp.LpVariable("EL", lowBound=0)

    obj1 = w_time * CT + w_eri * EL
    prob += obj1
    for i in tasks:
        prob += pulp.lpSum(x[i][s] for s in S) == 1
    for s in S:
        prob += pulp.lpSum(t[i] * x[i][s] for i in tasks) <= CT
        prob += pulp.lpSum(e[i] * x[i][s] for i in tasks) <= EL
    for a, b in precedence:
        prob += pulp.lpSum(s * x[a][s] for s in S) <= pulp.lpSum(s * x[b][s] for s in S)
    for s in S:
        for i in tasks:
            prob += x[i][s] <= y[s][ws[i]]
        for w in W:
            prob += y[s][w] <= pulp.lpSum(x[i][s] for i in tasks if ws[i] == w)
            prob += last[w] >= s * y[s][w]
            prob += first[w] <= s + m * (1 - y[s][w])
    for w in range(1, p_ws):
        if w in W and w + 1 in W:
            prob += last[w] <= first[w + 1]

    if warm_start:
        for i in tasks:
            for s in S:
                x[i][s].setInitialValue(1 if warm_start.get(i) == s else 0)

    solver = pulp.HiGHS(msg=msg, timeLimit=time_limit, mip_rel_gap=1e-6,
                        warmStart=bool(warm_start))
    t0 = time.perf_counter()
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    def read_assignment():
        a = {}
        for i in tasks:
            on = [s for s in S if (pulp.value(x[i][s]) or 0) > 0.5]
            if len(on) != 1:
                return None
            a[i] = on[0]
        return a

    assign = read_assignment()
    if assign is None:
        raise RuntimeError(f"MILP returned no solution (status {status}, m={m})")
    z1 = pulp.value(obj1)
    lex_done = False

    if lexicographic and status == "Optimal":
        # stage 2: keep the weighted peak at its optimum (relative tolerance), push both peaks down.
        # Start from the stage-1 solution; if stage 2 fails, the stage-1 solution is kept.
        prob += obj1 <= z1 * (1 + 1e-6) + 1e-6, "lex_keep_obj1"
        prob.setObjective(CT + EL)
        for i in tasks:
            for s in S:
                x[i][s].setInitialValue(1 if assign[i] == s else 0)
        prob.solve(pulp.HiGHS(msg=msg, timeLimit=time_limit, mip_rel_gap=1e-6, warmStart=True))
        if pulp.LpStatus[prob.status] != "Optimal":
            # HiGHS presolve can wrongly declare this stage infeasible on some instances
            # (checked against CBC); retry without presolve
            prob.solve(pulp.HiGHS(msg=msg, timeLimit=time_limit, mip_rel_gap=1e-6, warmStart=True,
                                  presolve="off"))
        stage2 = read_assignment()
        if pulp.LpStatus[prob.status] == "Optimal" and stage2 is not None:
            assign, lex_done = stage2, True

    loads_t = {s: sum(t[i] for i in tasks if assign[i] == s) for s in S}
    loads_e = {s: sum(e[i] for i in tasks if assign[i] == s) for s in S}
    ct, el = max(loads_t.values()), max(loads_e.values())
    return {"assignment": assign, "CT": ct, "EL": el, "objective": w_time * ct + w_eri * el,
            "time_loads": loads_t, "eri_loads": loads_e, "status": status,
            "lexicographic_stage2": lex_done, "seconds": time.perf_counter() - t0}
