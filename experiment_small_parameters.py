#!/usr/bin/env sage -python
"""Experiment of Section 6.1 (Table 4): Heuristics 1 and 2 on small and medium parameters.

Usage:
    sage -python experiment_small_parameters.py [--trials 25] [--q 4093] [--cores 1] [--out results_small]
Output: <out>.csv (one line per run) and <out>.tex (the LaTeX table of the paper).
"""

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algorithm1 as A1

SHAPES = [(n, n, n) for n in range(2, 11)] + [(12, 12, 12), (16, 16, 16), (20, 20, 20),
          (3, 4, 6), (4, 4, 6), (5, 4, 6), (7, 4, 6), (6, 4, 8), (4, 3, 5), (5, 6, 6), (2, 4, 4), (2, 3, 4)]
FAMILIES = ("2-sMCE", "sIMCE")


def admissible_signs(m, n):
    return [(ea, eb) for ea in ([1] + ([-1] if m % 2 == 0 else [])) for eb in ([1] + ([-1] if n % 2 == 0 else []))]


def run(task):
    fam, k, m, n, q, trial = task
    rng = random.Random((fam, k, m, n, q, trial).__repr__())
    ea, eb = rng.choice(admissible_signs(m, n))
    seed = rng.randrange(2 ** 62)
    G = A1.random_instance(fam, k, m, n, q, ea, eb, seed)
    out, info = A1.algorithm1(*G, m, n, ea, eb, seed=seed)
    predicted = k if ea * eb == 1 else (k + 1) // 2
    return dict(family=fam, k=k, m=m, n=n, q=q, trial=trial, eps_a=ea, eps_b=eb, d=info["d"],
                d_as_predicted=int(info["d"] == predicted), t=info["t"], kernel=info["kernel"],
                success=int(out is not None and A1.verify(*G, out)),
                time=round(info["time_step1"] + info["time_step2"] + info["time_step3"], 3))


def latex_table(rows, path):
    agg = {}
    for r in rows:
        a = agg.setdefault((r["family"], r["k"], r["m"], r["n"]), dict(N=0, dok=0, c1=0, ok=0))
        a["N"] += 1
        a["dok"] += r["d_as_predicted"]
        a["c1"] += r["kernel"] == 1
        a["ok"] += r["success"]
    lines = [r"\begin{tabular}{rrr l r r r r}", r"\toprule",
             r"$k$ & $m$ & $n$ & family & runs & $d$ as predicted & (C1) & success \\", r"\midrule"]
    for fi, fam in enumerate(FAMILIES):
        if fi:
            lines.append(r"\midrule")
        for (k, m, n) in SHAPES:
            a = agg.get((fam, k, m, n))
            if a:
                name = r"\twosmce" if fam == "2-sMCE" else r"$\simce$"
                lines.append(f"{k} & {m} & {n} & {name} & {a['N']} & {a['dok']} & {a['c1']} & {a['ok']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--q", type=int, default=4093)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--shapes", default="", help="comma-separated k:m:n list (default: all shapes of the paper)")
    a = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split(":")) for s in a.shapes.split(",")] if a.shapes else SHAPES
    tasks = [(f, k, m, n, a.q, tr) for f in FAMILIES for (k, m, n) in shapes for tr in range(a.trials)]
    tasks.sort(key=lambda t: -t[1] * t[2] * t[3])
    if a.cores > 1:
        with ProcessPoolExecutor(max_workers=a.cores) as ex:
            rows = list(ex.map(run, tasks))
    else:
        rows = [run(t) for t in tasks]
    with open("results_small.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    latex_table(rows, "results_small.tex")
    print(f"{len(rows)} runs: d as predicted in {sum(r['d_as_predicted'] for r in rows)}, "
          f"(C1) in {sum(r['kernel'] == 1 for r in rows)}, success in {sum(r['success'] for r in rows)}; "
          f"wrote results_small.csv and results_small.tex")


if __name__ == "__main__":
    main()
