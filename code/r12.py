#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import math
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc
import r7
import r11

PLAN = "SAP_addendum_R12.md v1.0"
NBOOT = 2000
MIN_HOSP_N = 50
SPECS = ("hosp", "nohosp")
OUTS = ("primary", "ic", "negctl")
SCHEMES = ("hosp", "pat")
NAMES = [f"{s}_{o}" for s in SPECS for o in OUTS]
HVARS = ("delirium", "screens", "rass_missing", "negctl")
RHO_NAMES = [f"rho_{v}" for v in HVARS]
SPEC_LAB = {"hosp": "Main analysis (with hospital indicators)", "nohosp": "Without hospital indicators (the MIMIC-IV model)"}
OUT_LAB = {"primary": "Delirium (as the main analysis, 168-h odds ratio)", "ic": "Onset hazard ratio taking assessment times into account (interval-censored, baseline in 3 segments)",
           "negctl": "Negative control: first glucose <70 mg/dL (168-h odds ratio)"}
HVAR_LAB = {"delirium": "Share with a first positive within 168 h (unweighted)", "screens": "Assessments per ICU day before the first positive",
            "rass_missing": "Share with RASS not recorded at time zero", "negctl": "Share with the negative control (unweighted)"}


def load_e(P):
    df, flow = asd.load_key(P, "c2e", stage2=True)
    d = asd.build(df, "e", 2, stage2=True)
    d.update(r11.timeline_arrays(d, df["stay_id"].values, asd.read_pq(r7.pf(P, "e", "r11tl"))))
    d["rass_na"] = ~np.isfinite(asd.num(df, "rass"))
    return d, flow


def spec_X(d, spec):
    if spec == "hosp":
        return d["X"], d["names"]
    keep = [j for j, nm in enumerate(d["names"]) if not nm.startswith("hosp_")]
    return d["X"][:, keep], [d["names"][j] for j in keep]


def estimates(d, cache):
    out = {}
    for s in SPECS:
        X, names = spec_X(d, s)
        w = r11.ow_of(d, cache, s, X, names)
        out[f"{s}_primary"] = r11.cif(d, w, d["t_del"])
        out[f"{s}_ic"] = r11.ic_fit(d, w, 3)
        out[f"{s}_negctl"] = r11.cif(d, w, d["t_hypo"])
    return out


def flat(res):
    vals = []
    for nm in NAMES:
        x = res[nm].get("HR" if nm.endswith("_ic") else "OR")
        vals.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)
    return np.array(vals)


def spearman(x, y):
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(x[ok].rank().corr(y[ok].rank()))


def hosp_summary(d):
    site = np.asarray(d["site"], float)
    hs, inv, cnt = np.unique(site, return_inverse=True, return_counts=True)
    big = cnt >= MIN_HOSP_N

    def mean_by(v):
        return np.bincount(inv, weights=np.asarray(v, float), minlength=hs.size) / cnt

    T = np.minimum(d["t_end"], float(r11.H))
    neg = np.isfinite(d["t_hypo"]) & (d["t_hypo"] <= T)
    share = mean_by(d["A"])
    vals = {"delirium": mean_by(d["ic_det"]),
            "screens": np.bincount(inv, weights=d["scr_any"], minlength=hs.size)
            / np.maximum(np.bincount(inv, weights=d["scr_days"], minlength=hs.size), 1e-12),
            "rass_missing": mean_by(d["rass_na"]), "negctl": mean_by(neg)}
    out = {"n_hosp": int(big.sum())}
    for k in HVARS:
        out[f"rho_{k}"] = spearman(share[big], vals[k][big])
    return out


def core_estimate(d, flow):
    cache = {}
    res = estimates(d, cache)
    hs = hosp_summary(d)
    return {"plan": PLAN, "flow": flow, "res": asd.to_json(res), "hosp": hs,
            "hosp_columns": [nm for nm in d["names"] if nm.startswith("hosp_")]}, cache


def estimate(args):
    t0 = time.time()
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    d, flow = load_e(P)
    est, cache = core_estimate(d, flow)
    res = est["res"]
    c2 = json.load(open(os.path.join(R, "ppi_estimates.json"), encoding="utf-8"))["per_db"]["e"]["analyses"]
    j11 = json.load(open(os.path.join(R, "r11.json"), encoding="utf-8"))["db"]["e"]
    checks = [("primary OR with hospital indicators vs ppi_estimates.json", res["hosp_primary"]["OR"], c2["primary"]["OR"]),
              ("negative-control OR with hospital indicators vs ppi_estimates.json", res["hosp_negctl"]["OR"], c2["negctl"]["OR"]),
              ("onset HR with hospital indicators vs r11.json", res["hosp_ic"]["HR"], j11["ic"]["HR"]),
              ("primary OR without hospital indicators vs R11 1c", res["nohosp_primary"]["OR"], j11["decomposition"]["blocks"]["hosp"]["minus"])]
    good = True
    for name, a, b in checks:
        ok = abs(round(a, 4) - b) < 1e-9
        good &= ok
        print(f"{'PASS' if ok else 'FAIL'} | {name}: {a:.4f} vs {b:.4f}", flush=True)
    if not good:
        asd.log_run(args.results, "r12 estimate: REPRODUCTION CHECK FAILED")
        raise SystemExit("the rebuilt eICU data differ from R6/R11: stop")
    json.dump(est, open(os.path.join(P, "r12_est.json"), "w"))
    np.savez(os.path.join(P, "r12_cache.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
    asd.status(f"r12 estimate done; {len(est['hosp_columns'])} hospital columns, {est['hosp']['n_hosp']} hospitals with >= {MIN_HOSP_N} patients", t0)
    asd.log_run(args.results, f"r12 estimate: ok, reproduction checks passed ({time.time() - t0:.0f} s)")


_G = {}


def names_of(scheme):
    return NAMES + (RHO_NAMES if scheme == "hosp" else [])


def draw(d, scheme, rep):
    idx, copy = r7.draw(d, "e" if scheme == "hosp" else "m", np.random.default_rng(asd.SEED + rep))
    b = asd.take(d, idx)
    if copy is not None:
        b["site"] = b["site"] * 1000 + copy
    return b


def _one(rep):
    d, scheme, cache = _G["d"], _G["scheme"], _G["cache"]
    try:
        b = draw(d, scheme, rep)
        v = flat(estimates(b, dict(cache)))
        if scheme == "hosp":
            h = hosp_summary(b)
            v = np.r_[v, [h[k] for k in RHO_NAMES]]
        return rep, v
    except Exception:
        return rep, None


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    d, _ = load_e(P)
    z = np.load(os.path.join(P, "r12_cache.npz"))
    _G.update(d=d, scheme=args.scheme, cache={k.replace("__", ":"): z[k] for k in z.files})
    reps = list(range(args.start, min(args.stop, NBOOT)))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, v in pool.imap_unordered(_one, reps, chunksize=5):
                out[rep] = v
    else:
        for rep in reps:
            out[rep] = _one(rep)[1]
    names = names_of(args.scheme)
    V = np.full((len(reps), len(names)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            V[i] = out[rep]
    np.savez(os.path.join(P, f"r12boot_{args.scheme}_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), v=V, names=np.array(names))
    failed = sum(1 for r in reps if out.get(r) is None)
    asd.status(f"r12 boot {args.scheme} {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r12 boot --scheme {args.scheme} --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def load_boots(P, scheme):
    files = sorted(glob.glob(os.path.join(P, f"r12boot_{scheme}_*.npz")))
    if not files:
        return None
    reps, vs, names = [], [], None
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        vs.append(z["v"])
        names = [str(x) for x in z["names"]]
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"names": names, "v": np.concatenate(vs)[first], "n": int(first.size)}


def raw_ci(B, name):
    if B is None or name not in B["names"]:
        return None, None
    v = B["v"][:, B["names"].index(name)]
    v = v[np.isfinite(v)]
    if v.size < asd.MIN_BOOT:
        return None, None
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def read_assoc(lo, hi, yes, no):
    if lo is None or hi is None:
        return "inconclusive"
    if hi < 1:
        return yes
    if lo > 1:
        return "opposite direction"
    return no


def read_negctl(lo, hi):
    if lo is None or hi is None:
        return "inconclusive"
    if hi < 1 or lo > 1:
        return "without hospital indicators a negative-control signal appears: differences between hospitals confound"
    return "negative control does not show confounding from dropping the hospital indicators"


def read_rho(lo, hi):
    if lo is None or hi is None:
        return "inconclusive"
    if hi < 0:
        return "hospitals using more PPI record less delirium: without adjustment for hospital, this difference between hospitals is counted in the difference between arms"
    if lo > 0:
        return "hospitals using more PPI record more delirium"
    return "no clear relation at hospital level"


def report(args):
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    est = json.load(open(os.path.join(P, "r12_est.json")))
    Bs = {s: load_boots(P, s) for s in SCHEMES}
    E = est["res"]
    out = {"plan": PLAN, "boot_reps": {s: (B["n"] if B else 0) for s, B in Bs.items()}, "flow": est["flow"],
           "hosp_columns": len(est["hosp_columns"]), "est": {}, "ratio": {}, "hosp": {"n_hosp": est["hosp"]["n_hosp"]}, "reading": {}}
    for nm in NAMES:
        x = E[nm]
        key = "HR" if nm.endswith("_ic") else "OR"
        ent = {key: x[key], "riskA": x.get("riskA"), "riskB": x.get("riskB")}
        for sch in SCHEMES:
            c = r11.with_ci(x[key], Bs[sch], nm, key)
            ent[sch] = {"lo": c.get("lo"), "hi": c.get("hi"), "se_log": c.get("se_log")}
        for k in ("eventsA", "eventsB", "nA", "nB", "n_inf_A", "n_inf_B"):
            if k in x:
                ent[k] = x[k]
        out["est"][nm] = ent
    for o in OUTS:
        key = "HR" if o == "ic" else "OR"
        for sch in SCHEMES:
            out["ratio"][f"{o}_{sch}"] = r11.paired_ratio(Bs[sch], f"nohosp_{o}", f"hosp_{o}", E[f"nohosp_{o}"][key], E[f"hosp_{o}"][key])
    for v in HVARS:
        lo, hi = raw_ci(Bs["hosp"], f"rho_{v}")
        out["hosp"][v] = {"rho": est["hosp"][f"rho_{v}"], "lo": lo, "hi": hi}
    rd = out["reading"]
    for o, lab in (("primary", ""), ("ic", "onset hazard ratio: ")):
        e = out["est"][f"nohosp_{o}"]
        rd[f"{o}_hosp"] = lab + read_assoc(e["hosp"]["lo"], e["hosp"]["hi"], "eICU also shows an association without hospital indicators", "eICU still shows no association without hospital indicators")
        rd[f"{o}_pat"] = (lab + "fully as in MIMIC-IV (no hospital indicators, patient-level resampling): "
                          + read_assoc(e["pat"]["lo"], e["pat"]["hi"], "eICU also shows an association", "eICU still shows no association"))
    e = out["est"]["nohosp_negctl"]
    rd["negctl"] = read_negctl(e["hosp"]["lo"], e["hosp"]["hi"])
    rd["hosp_rho"] = read_rho(out["hosp"]["delirium"]["lo"], out["hosp"]["delirium"]["hi"])
    masked = r11.mask_tree(asd.to_json(out))
    json.dump(asd.round_floats(masked), open(os.path.join(R, "r12.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r12.md"), "w", encoding="utf-8").write(md(masked))
    asd.log_run(args.results, "r12 report: ok")
    print("report written", flush=True)


def cell(e, sch):
    if not e or e.get(sch) is None or e[sch].get("lo") is None:
        return "—"
    return f"{r11.f2(e[sch]['lo'])}–{r11.f2(e[sch]['hi'])}"


def md(o):
    f2 = r11.f2
    L = ["# Revision R12 results: eICU with hospital handled as in MIMIC-IV (PPI vs famotidine)", "",
         f"Plan: {o['plan']}. Bootstrap: by hospital {o['boot_reps']['hosp']} replicates, by patient {o['boot_reps']['pat']} replicates (same seeds; the two models paired in every replicate). "
         f"Odds ratio <1: lower in the PPI arm. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", "",
         f"Analysis set: PPI {o['flow']['analysis_A']} / famotidine {o['flow']['analysis_B']} (outside overlap {o['flow']['outside_overlap']}); "
         f"the propensity model of the main analysis has {o['hosp_columns']} hospital indicator columns.", "",
         "## 1. Without hospital indicators", "",
         "| Outcome | Model | Estimate | 95% CI, resampling hospitals (eICU main analysis) | 95% CI, resampling patients (as in MIMIC-IV) |", "|---|---|---|---|---|"]
    for oc in OUTS:
        key = "HR" if oc == "ic" else "OR"
        for s in SPECS:
            e = o["est"][f"{s}_{oc}"]
            L.append(f"| {OUT_LAB[oc]} | {SPEC_LAB[s]} | {f2(e[key])} | {cell(e, 'hosp')} | {cell(e, 'pat')} |")
        rh, rp = o["ratio"].get(f"{oc}_hosp"), o["ratio"].get(f"{oc}_pat")
        L.append(f"| {OUT_LAB[oc]} | Without ÷ main analysis | {f2(rh['ratio']) if rh else '—'} | "
                 f"{(f2(rh['lo']) + '–' + f2(rh['hi'])) if rh else '—'} | {(f2(rp['lo']) + '–' + f2(rp['hi'])) if rp else '—'} |")
    L.append("")
    p = o["est"]["hosp_primary"]
    q = o["est"]["hosp_negctl"]
    L.append(f"- Events PPI / famotidine: delirium {p['eventsA']} / {p['eventsB']}; negative control {q['eventsA']} / {q['eventsB']}; "
             f"informative patients in the onset model {o['est']['hosp_ic']['n_inf_A']} / {o['est']['hosp_ic']['n_inf_B']}")
    a, b = o["est"]["hosp_primary"], o["est"]["nohosp_primary"]
    L.append(f"- Weighted 168-h risk PPI / famotidine: main analysis {f2(100 * a['riskA'], 1)}% / {f2(100 * a['riskB'], 1)}%; "
             f"without hospital indicators {f2(100 * b['riskA'], 1)}% / {f2(100 * b['riskB'], 1)}%")
    rd = o["reading"]
    L += [f"- Reading (written rule): {rd['primary_hosp']}; {rd['primary_pat']}", f"- Reading (written rule): {rd['ic_hosp']}; {rd['ic_pat']}",
          f"- Reading (written rule): {rd['negctl']}", "",
          "## 2. Hospital-level description", "",
          f"Hospitals with ≥{MIN_HOSP_N} patients in the analysis set: {o['hosp']['n_hosp']}; Spearman correlation of the PPI share of each hospital with each item below (percentile intervals from resampling hospitals).", "",
          "| PPI share with | Spearman ρ (95% CI) |", "|---|---|"]
    for v in HVARS:
        h = o["hosp"][v]
        L.append(f"| {HVAR_LAB[v]} | {f2(h['rho'])} ({f2(h['lo'])}–{f2(h['hi'])}) |" if h.get("lo") is not None
                 else f"| {HVAR_LAB[v]} | {f2(h['rho'])} |")
    L += ["", f"- Reading (written rule): {rd['hosp_rho']}", ""]
    return "\n".join(L)


def synth(n, rng, n_hosp=24, a_site=1.2, y_site=-0.8):
    site = rng.integers(0, n_hosp, n)
    u = rng.normal(size=n_hosp)[site]
    x = rng.normal(size=n)
    A = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + a_site * u + 0.5 * x)))).astype(int)
    t_end = rng.uniform(24, 240, n)
    onset = rng.exponential(1.0, n) / (0.004 * np.exp(y_site * u + 0.4 * x))
    hypo = rng.exponential(1.0, n) / (0.0015 * np.exp(y_site * u))
    T = np.minimum(t_end, 168.0)
    k = np.floor((T - 4.0) / 8.0).astype(int) + 1
    k = np.maximum(k, 0)
    sid = np.repeat(np.arange(n), k)
    h = 4.0 + 8.0 * (np.arange(sid.size) - np.repeat(np.cumsum(k) - k, k))
    pos = onset[sid] <= h
    tl = pd.DataFrame({"stay_id": sid, "h": h, "pos": pos.astype(int), "ok": np.ones(sid.size, int)})
    first = tl[tl["pos"] == 1].groupby("stay_id")["h"].min()
    t_del = np.full(n, np.nan)
    t_del[first.index.values] = first.values
    ref = np.bincount(site).argmax()
    cols = [x] + [(site == s).astype(float) for s in range(n_hosp) if s != ref]
    names = ["x"] + [f"hosp_{s}" for s in range(n_hosp) if s != ref]
    d = {"X": np.column_stack(cols), "names": names, "dropped": [], "A": A, "site": site.astype(float), "t_end": t_end,
         "end_death": np.zeros(n, bool), "t_dev": np.full(n, np.nan), "t_del": t_del, "t_hypo": np.where(hypo <= t_end, hypo, np.nan),
         "sub": {}, "rass_na": rng.random(n) < 0.3}
    d.update(r11.timeline_arrays(d, np.arange(n), tl))
    return d


def selftest(args):
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name}{' | ' + detail if detail else ''}", flush=True)

    rng = np.random.default_rng(12)
    d = {"X": np.arange(12.0).reshape(3, 4), "names": ["age_ge65", "hosp_H11", "vent", "hosp_small"]}
    X, nm = spec_X(d, "nohosp")
    check("without hospital indicators: exactly the hosp_ columns are dropped", nm == ["age_ge65", "vent"]
          and np.array_equal(X, d["X"][:, [0, 2]]) and spec_X(d, "hosp")[1] == d["names"])
    dd = synth(20000, rng)
    cache = {}
    r = estimates(dd, cache)
    check("hospital confounding only: main model near 1 (primary)", abs(math.log(r["hosp_primary"]["OR"])) < 0.1, f"{r['hosp_primary']['OR']:.3f}")
    check("hospital confounding only: without indicators the primary odds ratio is biased below 1", r["nohosp_primary"]["OR"] < 0.9,
          f"{r['nohosp_primary']['OR']:.3f}")
    check("hospital confounding only: the negative control shows the same bias without indicators, not with them",
          abs(math.log(r["hosp_negctl"]["OR"])) < 0.1 and r["nohosp_negctl"]["OR"] < 0.9,
          f"{r['hosp_negctl']['OR']:.3f} vs {r['nohosp_negctl']['OR']:.3f}")
    check("onset model finite under both models", all(np.isfinite(r[f"{s}_ic"]["HR"]) and r[f"{s}_ic"]["converged"] for s in SPECS),
          f"{r['hosp_ic']['HR']:.3f} / {r['nohosp_ic']['HR']:.3f}")
    good = True
    for sch, db in (("hosp", "e"), ("pat", "m")):
        b = draw(dd, sch, 7)
        ref = asd.resample(dd, db, np.random.default_rng(asd.SEED + 7))
        good &= np.array_equal(b["A"], ref["A"]) and np.array_equal(b["t_del"], ref["t_del"], equal_nan=True) \
            and np.array_equal(b["site"], ref["site"])
    check("bootstrap draws equal asd.resample for both schemes", good)
    _G.update(d=dd, scheme="hosp", cache=cache)
    v = _one(3)[1]
    b = draw(dd, "hosp", 3)
    ref = np.r_[flat(estimates(b, dict(cache))), [hosp_summary(b)[k] for k in RHO_NAMES]]
    check("replicate values in the declared order and reproducible", v is not None and len(v) == len(names_of("hosp"))
          and np.allclose(v, ref, equal_nan=True))
    n_each = [60, 60, 60, 20]
    site = np.repeat([1.0, 2.0, 3.0, 4.0], n_each)
    A = np.concatenate([(np.arange(m) < int(round(s * m))).astype(int) for m, s in zip(n_each, (0.2, 0.5, 0.8, 0.9))])
    det = np.concatenate([(np.arange(m) < int(round(s * m))) for m, s in zip(n_each, (0.3, 0.2, 0.1, 0.9))])
    toy = {"site": site, "A": A, "ic_det": det, "t_end": np.full(200, 100.0), "t_hypo": np.full(200, np.nan),
           "scr_any": np.repeat([1.0, 2.0, 3.0, 0.5], n_each), "scr_days": np.ones(200), "rass_na": np.zeros(200, bool)}
    hsum = hosp_summary(toy)
    check("hospital summary: size rule and Spearman correlations", hsum["n_hosp"] == 3 and abs(hsum["rho_delirium"] + 1) < 1e-12
          and abs(hsum["rho_screens"] - 1) < 1e-12, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in hsum.items()}))
    check("readings follow the written rules",
          read_assoc(0.7, 0.95, "y", "n") == "y" and read_assoc(0.8, 1.1, "y", "n") == "n" and read_assoc(1.05, 1.3, "y", "n") == "opposite direction"
          and read_negctl(0.9, 1.2).startswith("negative control does not") and read_negctl(0.6, 0.9).startswith("without hospital indicators a negative-control signal appears")
          and read_rho(-0.6, -0.1).startswith("hospitals using more PPI record less delirium") and read_rho(0.1, 0.5) == "hospitals using more PPI record more delirium"
          and read_rho(-0.3, 0.2) == "no clear relation at hospital level")
    import contextlib
    import io
    tmp = tempfile.mkdtemp(prefix="r12_self_")
    try:
        P, R = os.path.join(tmp, "p"), os.path.join(tmp, "r")
        os.makedirs(P)
        os.makedirs(R)
        small = synth(4000, rng)
        n = len(small["A"])
        flow = {"eligible": n, "A": int(small["A"].sum()), "B": int((small["A"] == 0).sum()), "other": 0, "outside_overlap": 0,
                "analysis_A": int(small["A"].sum()), "analysis_B": int((small["A"] == 0).sum())}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            est, cache = core_estimate(small, flow)
            json.dump(est, open(os.path.join(P, "r12_est.json"), "w"))
            for sch in SCHEMES:
                _G.update(d=small, scheme=sch, cache=cache)
                V = np.array([_one(rep)[1] for rep in range(60)])
                np.savez(os.path.join(P, f"r12boot_{sch}_0000_0060.npz"), reps=np.arange(60), v=V, names=np.array(names_of(sch)))
            report(argparse.Namespace(private=P, results=R))
        J = json.load(open(os.path.join(R, "r12.json"), encoding="utf-8"))
        mdtxt = open(os.path.join(R, "r12.md"), encoding="utf-8").read()
        check("end to end: report written with both sections and every estimate", "## 1." in mdtxt and "## 2." in mdtxt
              and all(J["est"][nm][s]["lo"] is not None for nm in NAMES for s in SCHEMES))
        cnt = []

        def walk(o, key=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, k)
            elif key in ("nA", "nB", "eventsA", "eventsB", "n_inf_A", "n_inf_B", "eligible", "analysis_A", "analysis_B"):
                cnt.append(o)
        walk(J)
        check("end to end: counts in the JSON are masked", bool(cnt) and all(isinstance(v, str) and (v == "<10" or int(v) % 5 == 0) for v in cnt),
              f"{len(cnt)} counts")
        check("end to end: every written reading present", all(k in J["reading"] for k in ("primary_hosp", "primary_pat", "ic_hosp", "ic_pat",
                                                                                          "negctl", "hosp_rho")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(res)}/{len(res)} passed", flush=True)
    sys.exit(0 if all(res) else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    subs = {c: sp.add_parser(c) for c in ("estimate", "boot", "report")}
    for s in subs.values():
        s.add_argument("--private", required=True)
        s.add_argument("--results", required=True)
    subs["boot"].add_argument("--scheme", choices=list(SCHEMES), required=True)
    subs["boot"].add_argument("--start", type=int, required=True)
    subs["boot"].add_argument("--stop", type=int, required=True)
    subs["boot"].add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"selftest": selftest, "estimate": estimate, "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
