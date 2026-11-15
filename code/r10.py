#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc

KEYS = ["c2m", "c2e"]
DBNAME = {"c2m": "MIMIC-IV", "c2e": "eICU"}
K = 5
FOLD_SEED = asd.SEED + 500000
GBM = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, early_stopping=False)
NAMES = ["mlps_primary", "mlps_negctl", "dr_primary", "dr_negctl"]
LIMIT_MIN = 60.0


def gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=0, **GBM)


def fit_predict(make, Xtr, ytr, Xte_list):
    if ytr.min() == ytr.max():
        return [np.full(len(x), float(ytr[0])) for x in Xte_list]
    m = make()
    m.fit(Xtr, ytr)
    return [m.predict_proba(x)[:, 1] for x in Xte_list]


def crossfit_ps(F, A, folds, make):
    e = np.empty(len(A))
    for k in range(K):
        tr, te = folds != k, folds == k
        if te.any():
            e[te] = fit_predict(make, F[tr], A[tr], [F[te]])[0]
    return np.clip(e, 1e-6, 1 - 1e-6)


def crossfit_outcome(F, A, Y, folds, make):
    n = len(A)
    m1, m0 = np.empty(n), np.empty(n)
    FA = np.column_stack([F, A])
    for k in range(K):
        tr, te = folds != k, folds == k
        if te.any():
            Fte, one = F[te], np.ones(int(te.sum()))
            m1[te], m0[te] = fit_predict(make, FA[tr], Y[tr], [np.column_stack([Fte, one]), np.column_stack([Fte, 0 * one])])
    return m1, m0


def aow(e, A, Y, m1, m0):
    h = e * (1 - e)
    w = np.where(A == 1, 1 - e, e)
    a1, a0 = A == 1, A == 0
    r1 = (h * m1).sum() / h.sum() + (w[a1] * (Y[a1] - m1[a1])).sum() / w[a1].sum()
    r0 = (h * m0).sum() / h.sum() + (w[a0] * (Y[a0] - m0[a0])).sum() / w[a0].sum()
    out = sc.effects(r1, r0)
    out["nA"], out["nB"] = int(a1.sum()), int(a0.sum())
    out["eventsA"], out["eventsB"] = int(Y[a1].sum()), int(Y[a0].sum())
    return out


def y168(d, t_ev):
    _, typ, _ = sc.follow_types(d["t_end"], d["end_death"], t_ev, disc_competing=True)
    return (typ == 1).astype(float)


def features(d):
    ex = d["extras"]
    sev = ex["sofa"] if "sofa" in ex else ex["apache"]
    return np.column_stack([d["X"], ex["age_years"], ex["egfr"], ex["h_icu"], sev, d["rass"]])


def flex(d, folds, make):
    F = features(d)
    A = d["A"]
    e = crossfit_ps(F, A, folds, make)
    ow = sc.overlap_w(e, A)
    res = {}
    for o, t in (("primary", d["t_del"]), ("negctl", d["t_hypo"])):
        res[f"mlps_{o}"] = asd.cif_contrast(d, ow, t)[168]
        Y = y168(d, t)
        m1, m0 = crossfit_outcome(F, A, Y, folds, make)
        res[f"dr_{o}"] = aow(e, A, Y, m1, m0)
    return res, e, ow


def auc(p, A):
    from scipy.stats import rankdata
    r = rankdata(p)
    n1 = float(A.sum())
    n0 = float(len(A) - n1)
    return float((r[A == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def draw(d, db, rng):
    n = len(d["A"])
    if db == "m":
        return rng.integers(0, n, n), None
    site = d["site"]
    hs = np.unique(site)
    pick = rng.choice(hs, size=hs.size, replace=True)
    rows = {h: np.nonzero(site == h)[0] for h in hs}
    idx = np.concatenate([rows[h] for h in pick])
    copy = np.concatenate([np.full(rows[h].size, i) for i, h in enumerate(pick)])
    return idx, copy


def main_folds(n):
    return np.random.default_rng(FOLD_SEED).integers(0, K, n)


def rep_folds(n, rep, idx):
    return np.random.default_rng(FOLD_SEED + 1 + rep).integers(0, K, n)[idx]


def load(P, key):
    db = key[2]
    df, _ = asd.load_key(P, key, stage2=True)
    d = asd.build(df, db, 2, stage2=True)
    d["rass"] = asd.num(df, "rass")
    return d, db


def estimate(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    for key in KEYS:
        d, db = load(P, key)
        A = d["A"]
        res, e, ow = flex(d, main_folds(len(A)), gbm)
        ps_lr = asd.ow_fit(d, {}, "full")
        smd = sc.smd_table(d["X"], d["names"], A, ow)
        diag = {"auc_logistic_in_sample": auc(ps_lr, A), "auc_gbm_cross_fitted": auc(e, A),
                "max_abs_smd_gbm_ow": max(abs(v) for v in smd.values() if np.isfinite(v)),
                "ess_gbm_ow": {"A": sc.ess(ow[A == 1]), "B": sc.ess(ow[A == 0])}, "features": int(features(d).shape[1])}
        json.dump({"res": asd.to_json(res), "diag": asd.to_json(diag)}, open(os.path.join(P, f"r10_est_{key}.json"), "w"))
        asd.status(f"r10 estimate {key} done", t0)
    asd.log_run(args.results, f"r10 estimate: ok ({time.time() - t0:.0f} s)")


_G = {}


def _one(rep):
    d, db = _G["d"], _G["db"]
    try:
        idx, copy = draw(d, db, np.random.default_rng(asd.SEED + rep))
        b = asd.take(d, idx)
        if copy is not None:
            b["site"] = b["site"] * 1000 + copy
        res, _, _ = flex(b, rep_folds(len(d["A"]), rep, idx), gbm)
    except Exception:
        return rep, None
    lor = [math.log(res[n]["OR"]) if np.isfinite(res[n].get("OR", np.nan)) and res[n]["OR"] > 0 else np.nan for n in NAMES]
    rd = [res[n]["RD"] if np.isfinite(res[n].get("RD", np.nan)) else np.nan for n in NAMES]
    return rep, (lor, rd)


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    d, db = load(P, args.key)
    _G.update(d=d, db=db)
    reps = list(range(args.start, args.stop))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, r in pool.imap_unordered(_one, reps, chunksize=2):
                out[rep] = r
    else:
        for rep in reps:
            out[rep] = _one(rep)[1]
    lor = np.full((len(reps), len(NAMES)), np.nan)
    rd = np.full((len(reps), len(NAMES)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            lor[i], rd[i] = out[rep]
    np.savez(os.path.join(P, f"r10_boot_{args.key}_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), lor=lor, rd=rd)
    failed = sum(1 for r in reps if out.get(r) is None)
    asd.status(f"r10 boot {args.key} {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r10 boot --key {args.key} --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def synth_d(n, p, rng, db):
    X = np.column_stack([rng.normal(size=(n, p // 2)), (rng.random((n, p - p // 2)) < 0.3).astype(float)])
    A = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.3 * X[:, 0] * X[:, p // 2])))).astype(int)
    t_end = rng.uniform(2, 240, n)
    d = {"X": X, "names": [f"x{j}" for j in range(p)], "dropped": [], "A": A, "t_end": t_end, "end_death": rng.random(n) < 0.08,
         "t_dev": np.full(n, np.nan), "sub": {}, "rass": np.where(rng.random(n) < 0.1, np.nan, rng.integers(-5, 3, n)).astype(float),
         "t_del": np.where(rng.random(n) < 0.25 - 0.05 * A, rng.uniform(0, 160, n), np.nan),
         "t_hypo": np.where(rng.random(n) < 0.08, rng.uniform(0, 160, n), np.nan),
         "site": (rng.integers(0, 24, n).astype(float) if db == "e" else None)}
    d["extras"] = {"age_years": rng.uniform(18, 95, n), "egfr": np.where(rng.random(n) < 0.05, np.nan, rng.uniform(10, 130, n)),
                   "h_icu": rng.uniform(0, 100, n), ("sofa" if db == "m" else "apache"): rng.uniform(0, 15, n)}
    return d


def timing(args):
    R = os.path.expanduser(args.results)
    c2 = json.load(open(os.path.join(R, "ppi_estimates.json"), encoding="utf-8"))["per_db"]
    rng = np.random.default_rng(1)
    total = 0.0
    lines = []
    for key in KEYS:
        db = key[2]
        f = c2[db]["flow"]
        n = int(f["analysis_A"]) + int(f["analysis_B"])
        d = synth_d(n, 60, rng, db)
        _G.update(d=d, db=db)
        t0 = time.time()
        for rep in range(2):
            _one(rep)
        per = (time.time() - t0) / 2
        total += per / 2 / 60
        lines.append(f"{key}: n {n} (masked), {per:.1f} s per replicate on one core")
    reps = next((r for r in (2000, 1000, 500, 200) if r * total <= LIMIT_MIN), 200)
    lines.append(f"projected minutes for 2000 / 1000 / 500 / 200 replicates (both databases, two workers): "
                 f"{2000 * total:.0f} / {1000 * total:.0f} / {500 * total:.0f} / {200 * total:.0f}; limit {LIMIT_MIN:.0f}")
    lines.append(f"replicates: {reps}")
    open(os.path.join(R, "r10_timing.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def load_boot(P, key):
    files = sorted(glob.glob(os.path.join(P, f"r10_boot_{key}_*.npz")))
    if not files:
        return None
    reps, lor, rd = [], [], []
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        lor.append(z["lor"])
        rd.append(z["rd"])
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"lor": np.concatenate(lor)[first], "rd": np.concatenate(rd)[first], "n": int(first.size)}


def summarise(res, bt):
    out = {}
    for i, n in enumerate(NAMES):
        e = dict(res[n])
        if bt is not None:
            lor, rd = bt["lor"][:, i], bt["rd"][:, i]
            lor, rd = lor[np.isfinite(lor)], rd[np.isfinite(rd)]
            if lor.size >= asd.MIN_BOOT:
                e["OR_lo"], e["OR_hi"] = math.exp(np.percentile(lor, 2.5)), math.exp(np.percentile(lor, 97.5))
                e["se_logOR"], e["boot_ok"] = float(lor.std(ddof=1)), int(lor.size)
            if rd.size >= asd.MIN_BOOT:
                e["RD_lo"], e["RD_hi"] = float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5))
        out[n] = e
    return out


def pool(a, b):
    ok = [x for x in (a, b) if x and x.get("se_logOR") and x.get("OR", 0) > 0]
    if len(ok) < 2:
        return None
    m, se, Q, I2, k = sc.pool_fe([math.log(x["OR"]) for x in ok], [x["se_logOR"] for x in ok])
    return {"OR": math.exp(m), "OR_lo": math.exp(m - 1.96 * se), "OR_hi": math.exp(m + 1.96 * se), "I2": I2, "Q": Q}


def report(args):
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    c2 = json.load(open(os.path.join(R, "ppi_estimates.json"), encoding="utf-8"))
    main = {key: {o: c2["per_db"][key[2]]["analyses"][o] for o in ("primary", "negctl")} for key in KEYS}
    main_pooled = {o: c2["pooled"][o] for o in ("primary", "negctl")}
    out = {"plan": "SAP_addendum_R10.md v1.0", "boot_reps": {}, "per_key": {}, "diag": {}, "pooled": {}, "main": main,
           "main_pooled": main_pooled, "consistent": {}, "reading": {}}
    for key in KEYS:
        est = json.load(open(os.path.join(P, f"r10_est_{key}.json")))
        bt = load_boot(P, key)
        out["boot_reps"][key] = bt["n"] if bt else 0
        out["per_key"][key] = summarise(est["res"], bt)
        out["diag"][key] = est["diag"]
    for n in NAMES:
        out["pooled"][n] = pool(out["per_key"]["c2m"][n], out["per_key"]["c2e"][n])
    for key in KEYS:
        for n in NAMES:
            o = n.split("_")[1]
            m, e = main[key][o], out["per_key"][key][n]
            out["consistent"][f"{key}:{n}"] = bool(m["OR_lo"] <= e["OR"] <= m["OR_hi"])
    for n in NAMES:
        p = out["pooled"][n]
        if p:
            out["reading"][f"pooled:{n}"] = asd.interpret(p["OR"], p["OR_lo"], p["OR_hi"])
    mp = main_pooled["primary"]
    out["reading"]["pooled:main_primary"] = asd.interpret(mp["OR"], mp["OR_lo"], mp["OR_hi"])
    same = all(out["consistent"][f"{k}:{n}"] for k in KEYS for n in ("mlps_primary", "dr_primary")) and all(
        out["reading"].get(f"pooled:{n}") == out["reading"]["pooled:main_primary"] for n in ("mlps_primary", "dr_primary"))
    out["reading"]["overall"] = "conclusion does not depend on the form of the propensity model" if same else "conclusion changes with the form of the model"
    masked = mask(out)
    json.dump(asd.round_floats(masked), open(os.path.join(R, "r10.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r10.md"), "w", encoding="utf-8").write(md(masked))
    asd.log_run(args.results, "r10 report: ok")
    print("report written", flush=True)


def mask(o, key=""):
    if isinstance(o, dict):
        return {k: mask(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [mask(v, key) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)) and key in ("eventsA", "eventsB", "nA", "nB"):
        return sc.mask_n(int(o))
    return o


def orci(e):
    if not e or e.get("OR") is None:
        return "—"
    return asd.fmt(e["OR"]) + (f" ({asd.fmt(e['OR_lo'])}–{asd.fmt(e['OR_hi'])})" if e.get("OR_lo") is not None else "")


def md(o):
    lab = {"main": "Main analysis (logistic propensity score, overlap weighting)", "mlps": "2a gradient-boosted propensity score, overlap weighting", "dr": "2b doubly robust (augmented overlap weighting, gradient boosting)"}
    L = ["# Revision R10 results: PPI comparison main analysis with flexible models (MIMIC-IV, eICU)", "",
         f"Plan: {o['plan']}. Bootstrap replicates: {', '.join(f'{DBNAME[k]} {v}' for k, v in o['boot_reps'].items())} (MIMIC-IV by patient, eICU by hospital). "
         "Odds ratios PPI vs famotidine, 168 hours, ICU discharge alive and death as competing events. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", "",
         "| Outcome | Estimate | MIMIC-IV | eICU | Pooled (I²) |", "|---|---|---|---|---|"]
    for oc, ot in (("primary", "delirium (primary outcome)"), ("negctl", "negative control: glucose <70 mg/dL")):
        mp = o["main_pooled"][oc]
        L.append(f"| {ot} | {lab['main']} | {orci(o['main']['c2m'][oc])} | {orci(o['main']['c2e'][oc])} | "
                 f"{orci(mp)}{'; I² {:.0f}%'.format(100 * mp['I2']) if mp.get('I2') is not None else ''} |")
        for est in ("mlps", "dr"):
            n = f"{est}_{oc}"
            p = o["pooled"].get(n)
            cm, ce = o["consistent"][f"c2m:{n}"], o["consistent"][f"c2e:{n}"]
            L.append(f"| {ot} | {lab[est]} | {orci(o['per_key']['c2m'][n])}{'' if cm else ' (outside the main-analysis interval)'} | "
                     f"{orci(o['per_key']['c2e'][n])}{'' if ce else ' (outside the main-analysis interval)'} | "
                     f"{orci(p)}{'; I² {:.0f}%'.format(100 * p['I2']) if p else ''} |")
    L += ["", f"- Reading (written rule): {o['reading']['overall']}. Pooled primary outcome: main analysis {o['reading']['pooled:main_primary']}; "
          f"2a {o['reading'].get('pooled:mlps_primary', '')}; 2b {o['reading'].get('pooled:dr_primary', '')}", ""]
    for k in KEYS:
        g = o["diag"][k]
        L.append(f"- {DBNAME[k]}: propensity score AUC, logistic regression (in sample) {g['auc_logistic_in_sample']:.3f}, gradient boosting (cross-fitted) {g['auc_gbm_cross_fitted']:.3f}; "
                 f"2a largest |SMD| after weighting {g['max_abs_smd_gbm_ow']:.3f}, effective sample size A {g['ess_gbm_ow']['A']:.0f}, B {g['ess_gbm_ow']['B']:.0f}; {g['features']} features")
    L.append("")
    return "\n".join(L)


class _Logit:

    def fit(self, X, y):
        self.beta = sc.logit_fit(np.nan_to_num(X), y)
        return self

    def predict_proba(self, X):
        p = sc.logit_pred(self.beta, np.nan_to_num(X))
        return np.column_stack([1 - p, p])


def selftest(args):
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name}{' | ' + detail if detail else ''}", flush=True)

    rng = np.random.default_rng(3)
    d = synth_d(3000, 10, rng, "m")
    w = rng.uniform(0.1, 1, 3000)
    e = asd.cif_contrast(d, w, d["t_del"])[168]
    Y, A = y168(d, d["t_del"]), d["A"]
    ok = abs(e["riskA"] - (w * Y)[A == 1].sum() / w[A == 1].sum()) < 1e-12 and abs(e["riskB"] - (w * Y)[A == 0].sum() / w[A == 0].sum()) < 1e-12
    check("binary outcome by 168 h equals the weighted Aalen-Johansen risk", ok)
    n = 40000
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), (rng.random(n) < 0.4).astype(float)
    ps = 1 / (1 + np.exp(-(0.5 * x1 - 0.5 * x2 + 0.8 * x1 * x3)))
    A = (rng.random(n) < ps).astype(int)

    def mu(a):
        return 1 / (1 + np.exp(-(-1.5 + 0.6 * x1 + 0.4 * x2 - 0.4 * a + 0.5 * x3)))

    Y = (rng.random(n) < np.where(A == 1, mu(1), mu(0))).astype(float)
    h = ps * (1 - ps)
    truth = sc.effects((h * mu(1)).sum() / h.sum(), (h * mu(0)).sum() / h.sum())["OR"]
    r = aow(ps, A, Y, mu(1), mu(0))
    check("augmented OW, both models true", abs(math.log(r["OR"]) - math.log(truth)) < 0.05, f"{r['OR']:.3f} vs {truth:.3f}")
    r = aow(ps, A, Y, np.full(n, Y.mean()), np.full(n, Y.mean()))
    check("augmented OW, outcome model wrong (constant), propensity true", abs(math.log(r["OR"]) - math.log(truth)) < 0.06, f"{r['OR']:.3f} vs {truth:.3f}")
    e0 = np.full(n, 0.5)
    t0_ = sc.effects(mu(1).mean(), mu(0).mean())["OR"]
    r = aow(e0, A, Y, mu(1), mu(0))
    check("augmented OW, propensity wrong (constant), outcome true: the population defined by the wrong score",
          abs(math.log(r["OR"]) - math.log(t0_)) < 0.05, f"{r['OR']:.3f} vs {t0_:.3f}")
    F = np.column_stack([x1, x2, x3])
    folds = np.random.default_rng(5).integers(0, K, n)
    try:
        eh = crossfit_ps(F, A, folds, gbm)
        m1, m0 = crossfit_outcome(F, A, Y, folds, gbm)
        r = aow(eh, A, Y, m1, m0)
        hh = eh * (1 - eh)
        target = sc.effects((hh * mu(1)).sum() / hh.sum(), (hh * mu(0)).sum() / hh.sum())["OR"]
        check("augmented OW with cross-fitted gradient boosting recovers the overlap-population OR",
              abs(math.log(r["OR"]) - math.log(target)) < 0.08, f"{r['OR']:.3f} vs {target:.3f}")
        dd = synth_d(4000, 20, rng, "e")
        out, _, _ = flex(dd, main_folds(4000), gbm)
        check("flex on a synthetic analysis set: every estimate finite", all(np.isfinite(out[k]["OR"]) for k in NAMES))
    except ImportError:
        check("scikit-learn available", False, "install the wheels first")
    for db in ("m", "e"):
        dd = synth_d(2000, 6, rng, db)
        idx, copy = draw(dd, db, np.random.default_rng(asd.SEED + 7))
        b = asd.take(dd, idx)
        if copy is not None:
            b["site"] = b["site"] * 1000 + copy
        ref = asd.resample(dd, db, np.random.default_rng(asd.SEED + 7))
        check(f"draws equal asd.resample ({db})", np.array_equal(b["X"], ref["X"]) and (db == "m" or np.array_equal(b["site"], ref["site"])))
        f = rep_folds(2000, 7, idx)
        same = all(len(set(f[idx == i])) <= 1 for i in np.unique(idx)[:300])
        check(f"copies of a drawn patient share a fold ({db})", same)
    dd = synth_d(3000, 8, rng, "m")
    out, _, _ = flex(dd, main_folds(3000), _Logit)
    check("pipeline with a logistic stand-in: every estimate finite", all(np.isfinite(out[k]["OR"]) for k in NAMES))
    print(f"{sum(res)}/{len(res)} passed")
    sys.exit(0 if all(res) else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    s = sp.add_parser("timing")
    s.add_argument("--results", required=True)
    subs = {c: sp.add_parser(c) for c in ("estimate", "boot", "report")}
    for s in subs.values():
        s.add_argument("--private", required=True)
        s.add_argument("--results", required=True)
    subs["boot"].add_argument("--key", choices=KEYS, required=True)
    subs["boot"].add_argument("--start", type=int, required=True)
    subs["boot"].add_argument("--stop", type=int, required=True)
    subs["boot"].add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"selftest": selftest, "timing": timing, "estimate": estimate, "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
