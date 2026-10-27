#!/usr/bin/env sage -python
"""Experiment of Section 6.2 (Figure 3): failure probabilities of Heuristics 1 and 2 as a function of q.

Usage:
    sage -python experiment_field_size.py [--k 10] [--trials 500] [--cores 1] [--out results_q]
Output: <out>.csv (one line per point) and <out>.tex (the pgfplots figure of the paper).
"""

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algorithm1 as A1
from sage.all import primes

PRIME_BOUND = 257
PRIMES = [int(p) for p in primes(3, PRIME_BOUND + 1)]
FAMILIES = ("2-sMCE", "sIMCE")
SIGNS = [(1, 1), (-1, 1)]
ZERO_LEVEL = 3e-4   # zero counts are drawn on the baseline labelled 0


def run_batch(task):
    fam, k, q, ea, eb, start, count = task
    predicted = k if ea * eb == 1 else (k + 1) // 2
    h1_fail = h2_fail = success = 0
    d_hist = {}
    for trial in range(start, start + count):
        seed = random.Random((fam, k, q, ea, eb, trial).__repr__()).randrange(2 ** 62)
        G = A1.random_instance(fam, k, k, k, q, ea, eb, seed)
        out, info = A1.algorithm1(*G, k, k, ea, eb, seed=seed)
        h1_fail += info["d"] != predicted
        h2_fail += info["kernel"] != 1
        success += out is not None and A1.verify(*G, out)
        d_hist[info["d"]] = d_hist.get(info["d"], 0) + 1
    return dict(family=fam, k=k, q=q, eps_a=ea, eps_b=eb, trials=count, heur1_fail=h1_fail, heur2_fail=h2_fail,
                success=success, d_hist=";".join(f"{d}:{c}" for d, c in sorted(d_hist.items())))


def pgfplots_figure(rows, k, per, path):
    L = [r"\begin{tikzpicture}",
         r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.3cm},",
         r"    width=0.5\textwidth, height=5.8cm, xmode=log, ymode=log, xlabel={$q$},",
         r"    ymin=2e-4, ymax=1, xmin=1, xmax=1300, ytick={0.0003,0.001,0.01,0.1,1}, yticklabels={$0$,$10^{-3}$,$10^{-2}$,$10^{-1}$,$10^{0}$}, xtick={1,10,100,1000}, grid=major, grid style={color=black!10},",
         r"    axis line style={color=black!50}, tick label style={font=\scriptsize}, label style={font=\footnotesize},",
         r"    title style={font=\footnotesize, yshift=-2pt}, legend cell align=left,",
         r"    legend style={font=\tiny, at={(0.97,0.97)}, anchor=north east, draw=none, fill=white, fill opacity=0.95, text opacity=1, row sep=-2pt},",
         r"    every axis plot/.append style={line width=0.6pt}]"]
    styles = {(1, "heur2_fail"): ("plotA, mark=o, mark options={fill=white}", r"Heuristic~2, $\varepsilon=+1$"),
              (1, "heur1_fail"): ("plotA, densely dotted, mark=*", r"Heuristic~1, $\varepsilon=+1$"),
              (-1, "heur2_fail"): ("plotB, mark=square, mark options={fill=white}", r"Heuristic~2, $\varepsilon=-1$"),
              (-1, "heur1_fail"): ("plotB, densely dotted, mark=square*", r"Heuristic~1, $\varepsilon=-1$")}
    for fam, name in (("2-sMCE", r"$\twosmce$"), ("sIMCE", r"$\simce$")):
        L.append(rf"\nextgroupplot[title={{{name}, $k=m=n={k}$}}" + (r", ylabel={probability}" if fam == "2-sMCE" else "") + "]")
        for (ea, _), what in [((1, 1), "heur2_fail"), ((1, 1), "heur1_fail"), ((-1, 1), "heur2_fail"), ((-1, 1), "heur1_fail")]:
            style, label = styles[(ea, what)]
            pts = [(r["q"], r[what] / r["trials"]) for r in sorted(rows, key=lambda r: r["q"])
                   if r["family"] == fam and r["eps_a"] == ea]
            nz, zero = [x for x in pts if x[1] > 0], [(q, ZERO_LEVEL) for q, v in pts if v == 0]
            if nz:
                L.append(rf"\addplot[{style}] coordinates {{" + " ".join(f"({q},{v:.6g})" for q, v in nz) + "};")
                L.append(rf"\addlegendentry{{{label}}}")
            if zero:   # zero counts: unconnected markers on the baseline, no legend entry
                L.append(rf"\addplot[{style}, only marks, forget plot] coordinates {{" + " ".join(f"({q},{v})" for q, v in zero) + "};")
        L.append(r"\addplot[color=black!70, thick, dashed, line width=0.9pt, no marks] coordinates {" + " ".join(f"({q},{1 / q:.6g})" for q in sorted({r["q"] for r in rows})) + "};")
        L.append(r"\addlegendentry{$1/q$}")
    L += [r"\end{groupplot}", r"\end{tikzpicture}",
          r"% requires \usepackage{pgfplots}, \usepgfplotslibrary{groupplots} and the styles",
          r"% \pgfplotsset{plotA/.style={color=plotblue, mark size=1.4pt}, plotB/.style={color=plotorange, mark size=1.4pt}}"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--cores", type=int, default=1)
    a = ap.parse_args()
    primes = PRIMES
    tasks = [(f, a.k, q, ea, eb, s, min(a.batch, a.trials - s)) for f in FAMILIES for q in primes
             for (ea, eb) in SIGNS for s in range(0, a.trials, a.batch)]
    if a.cores > 1:
        with ProcessPoolExecutor(max_workers=a.cores) as ex:
            parts = list(ex.map(run_batch, tasks))
    else:
        parts = [run_batch(t) for t in tasks]
    agg = {}
    for p in parts:
        key = (p["family"], p["q"], p["eps_a"], p["eps_b"])
        r = agg.setdefault(key, dict(p, trials=0, heur1_fail=0, heur2_fail=0, success=0, d_hist={}))
        for f in ("trials", "heur1_fail", "heur2_fail", "success"):
            r[f] += p[f]
        for item in p["d_hist"].split(";"):
            d, c = item.split(":")
            r["d_hist"][int(d)] = r["d_hist"].get(int(d), 0) + int(c)
    rows = []
    for key in sorted(agg, key=lambda x: (x[0], x[2], x[1])):
        r = dict(agg[key])
        r["d_hist"] = ";".join(f"{d}:{c}" for d, c in sorted(r["d_hist"].items()))
        rows.append(r)
    with open("results_q.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pgfplots_figure(rows, a.k, a.trials, "results_q.tex")
    worst1 = max(r["q"] * r["heur1_fail"] / r["trials"] for r in rows)
    worst2 = max(r["q"] * r["heur2_fail"] / r["trials"] for r in rows)
    bad2 = [r["q"] for r in rows if r["heur2_fail"]]
    print(f"{sum(r['trials'] for r in rows)} runs; Heuristic 2 fails only for q <= {max(bad2) if bad2 else None}, "
          f"max q*P[Heur. 2 fails] = {worst2:.2f}, max q*P[Heur. 1 fails] = {worst1:.2f}; wrote results_q.csv and results_q.tex")


if __name__ == "__main__":
    main()
