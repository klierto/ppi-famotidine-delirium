#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

sys.dont_write_bytecode = True
import argparse
import datetime
import glob
import hashlib
import json
import math
import subprocess
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import asd
import explore_ppi as e1
import r7
import r10
import r11
import r12
import r14
import r16
import r16_a16
import stats_core as sc

PLAN = "SAP_addendum_R18.md v1.0"
SUB = "r18"
SEED = asd.SEED
MIN_BOOT = asd.MIN_BOOT
FORBIDDEN = (ROOT, os.environ.get("DATA_DIR", "data"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(path, folder):
    a, b = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def priv(args):
    P = os.path.abspath(os.path.expanduser(args.private))
    if any(inside(P, f) for f in FORBIDDEN):
        raise SystemExit("the private folder must be outside the repository and the databases")
    if not os.path.isdir(P):
        raise SystemExit("private folder not found")
    os.makedirs(os.path.join(P, SUB, "logs"), exist_ok=True)
    return P


def res_dir(args):
    R = os.path.abspath(os.path.expanduser(args.results))
    if os.path.normcase(R) != os.path.normcase(os.path.join(ROOT, "results")):
        raise SystemExit("--results must be the project's results folder")
    return R


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def pr(P, *parts):
    return os.path.join(P, SUB, *parts)


def record(P, step, t0, start, extra=None):
    f = pr(P, "timing.json")
    T = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {"steps": []}
    T["steps"].append(dict({"step": step, "start": start, "end": now_iso(), "seconds": round(time.time() - t0, 1)}, **(extra or {})))
    json.dump(T, open(f, "w", encoding="utf-8"), indent=1)


def say(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def lg(x):
    return math.log(x) if (x is not None and isinstance(x, (int, float, np.floating)) and np.isfinite(x) and x > 0) else np.nan


def lor(e, k="OR"):
    return lg(e.get(k)) if isinstance(e, dict) else np.nan


def draw(d, scheme, rep):
    return r12.draw(d, scheme, rep)


ASD_COLS = ["primary", "del72", "negctl", "rass3", "egfr60", "age65", "hyp", "pp", "iptw", "cc", "crude", "death"]


def asd_load(P, est=None):
    df, flow = asd.load_key(P, "c2e", stage2=True)
    return {"d": asd.build(df, "e", 2, stage2=True), "flow": flow}


def asd_point(st):
    cache = {}
    return asd.run_stage2(st["d"], 2, "e", cache), cache


def asd_vec(st, scheme, rep, cache):
    res = asd.run_stage2(draw(st["d"], scheme, rep), 2, "e", dict(cache))
    return [lor(res.get(n)) for n in ASD_COLS]


R7_COLS = ["m0_primary", "m0_negctl", "m1_primary", "m2_primary", "m2_rass3", "m1_negctl", "m2_negctl", "px_cam", "px_icu"]


def r7_load(P, est=None):
    df, flow, dd = r7.load_dd(P, "c2e")
    st = {"dd": dd, "d": dd["m0"], "flow": flow}
    if est is None:
        st["df"] = df
    return st


def r7_point(st):
    cache = {}
    res = r7.analyses(st["dd"], "c2e", cache)
    for m, d in st["dd"].items():
        r7.balance(d, cache, m)
    if "df" in st:
        r7.describe_new(st["df"], st["dd"], cache)
    return res, cache


def r7_vec(st, scheme, rep, cache):
    res = r7.replicate(st["dd"], "c2e", "e" if scheme == "hosp" else "m", rep, cache)
    return [lor(res.get(n)) for n in R7_COLS]


R11_COLS = ["primary", "negctl", "ic", "m3_primary", "m3_negctl", "ic6"]


def r11_load(P, est=None):
    d, _, flow = r11.load_ppi(P, "e")
    return {"d": d, "flow": flow}


def r11_point(st):
    cache = {}
    return r11.analyses_all(st["d"], "e", cache), cache


def r11_vec(st, scheme, rep, cache):
    v = r11.flat(r11.analyses_all(draw(st["d"], scheme, rep), "e", dict(cache)), "e")
    names = r11.flat_names("e")
    return [v[names.index(n)] for n in R11_COLS]


R10_COLS = list(r10.NAMES)


def r10_load(P, est=None):
    d, _ = r10.load(P, "c2e")
    return {"d": d}


def r10_point(st):
    res, _, _ = r10.flex(st["d"], r10.main_folds(len(st["d"]["A"])), r10.gbm)
    return res, {}


def r10_vec(st, scheme, rep, cache):
    d = st["d"]
    n = len(d["A"])
    idx, copy = r10.draw(d, "e" if scheme == "hosp" else "m", np.random.default_rng(SEED + rep))
    b = asd.take(d, idx)
    if copy is not None:
        b["site"] = b["site"] * 1000 + copy
    res, _, _ = r10.flex(b, r10.rep_folds(n, rep, idx), r10.gbm)
    return [lor(res.get(nm)) for nm in R10_COLS]


A16_COLS = ["px_icu:A", "px_icu:B"]


def a16_load(P, est=None):
    d, flow = r16_a16.load(P, "e", "nodel")
    return {"d": d, "flow": flow}


def a16_point(st):
    cache = {}
    return r16_a16.est(st["d"], "e", cache), cache


def a16_vec(st, scheme, rep, cache):
    r = r16_a16.est(draw(st["d"], scheme, rep), "e", dict(cache))
    return [r["px_icu"]["riskA"], r["px_icu"]["riskB"]]


C1_NAMES = ["primary", "kadj_primary", "kexc_primary", "bleedexc_primary"]
C1_COLS = [f"{n}:{a}" for n in C1_NAMES for a in ("A", "B")]


def c1_load(P, est=None):
    d, flow = r16.load_key(P, "c1e")
    return {"d": d, "flow": flow}


def c1_point(st):
    cache = {}
    return r16.est_key(st["d"], "c1e", cache), cache


def c1_vec(st, scheme, rep, cache):
    res = r16.est_key(r16.draw(st["d"], "e", "main" if scheme == "hosp" else "pat", rep), "c1e", dict(cache))
    return [res[n][k] for n in C1_NAMES for k in ("riskA", "riskB")]


E1_COLS = ["primary", "vent:no", "vent:yes", "vent:yes:ratio", "rass:deep:ratio", "std_own", "std_other"]


def e1_load(P, est=None):
    d, flow = e1.load(P, "e")
    st = {"d": d, "flow": flow}
    if est is not None:
        st["shares_m"] = est["shares_m"]
    else:
        dm, _ = e1.load(P, "m")
        st["shares_m"] = e1.rass_shares(dm)
    return st


def e1_point(st):
    res, _ = e1.all_estimates(st["d"], "e", st["shares_m"])
    return res, {}


def e1_vec(st, scheme, rep, cache):
    res, _ = e1.all_estimates(draw(st["d"], scheme, rep), "e", st["shares_m"])
    names, v = e1.flat(res, "e")
    return [v[names.index(n)] for n in E1_COLS]


R14_COLS = ["main_primary", "main_ic", "pre_primary", "strat_pre_0", "strat_pre_1"]


def r14_load(P, est=None):
    d, flow = r14.load_e(P)
    return {"d": d, "flow": flow}


def r14_point(st):
    cache = {}
    return r14.estimates(st["d"], cache), cache


def r14_vec(st, scheme, rep, cache):
    v = r14.flat(r14.estimates(draw(st["d"], scheme, rep), dict(cache)))
    return [v[r14.NAMES.index(n)] for n in R14_COLS]


PIPES = {
    "asd": dict(script="asd.py", nboot=2000, chunk=125, cols=ASD_COLS, load=asd_load, point=asd_point, vec=asd_vec),
    "r7": dict(script="r7.py", nboot=2000, chunk=125, cols=R7_COLS, load=r7_load, point=r7_point, vec=r7_vec),
    "r11": dict(script="r11.py", nboot=2000, chunk=125, cols=R11_COLS, load=r11_load, point=r11_point, vec=r11_vec),
    "r10": dict(script="r10.py", nboot=1000, chunk=50, cols=R10_COLS, load=r10_load, point=r10_point, vec=r10_vec),
    "a16": dict(script="r16_a16.py", nboot=2000, chunk=125, cols=A16_COLS, load=a16_load, point=a16_point, vec=a16_vec),
    "c1": dict(script="r16.py", nboot=2000, chunk=125, cols=C1_COLS, load=c1_load, point=c1_point, vec=c1_vec),
    "e1": dict(script="explore_ppi.py", nboot=2000, chunk=125, cols=E1_COLS, load=e1_load, point=e1_point, vec=e1_vec),
    "r14": dict(script="r14.py", nboot=2000, chunk=125, cols=R14_COLS, load=r14_load, point=r14_point, vec=r14_vec),
}
ORDER = ["r10", "r14", "r11", "asd", "e1", "r7", "c1", "a16"]


def load_stored(R):
    J = lambda f: json.load(open(os.path.join(R, f), encoding="utf-8"))
    r16j = J("r16.json")
    return {"c2": J("ppi_estimates.json")["per_db"]["e"]["analyses"], "r7": J("r7.json"), "r10": J("r10.json")["per_key"]["c2e"],
            "r11": J("r11.json")["db"]["e"], "a16": J("r16_a16.json")["db"]["e"]["nodel"]["est"], "c1": r16j["famdose"]["e"]["est"],
            "e1": J("ppi_explore_E1.json")["db"], "r14": J("r14.json"), "pat": r16j["db"]["e"]["est_pat"], "icpat": r16j["db"]["e"]["ic_pat"]}


def _or(e, k="OR"):
    return (e[k], e[k + "_lo"], e[k + "_hi"])


def _lohi(e, k="OR"):
    return (e[k], e["lo"], e["hi"])


def _ci(e, k="OR", c="CI"):
    return (e[k], e[c][0], e[c][1])


def it(i, pipe, kind, label, where, key, stored, point, spec, n=None, dp=4, tol=None, check=False):
    return dict(id=i, pipe=pipe, kind=kind, label=label, where=where, script=PIPES[pipe]["script"], key=key, stored=stored, point=point,
                spec=spec, n=n, dp=dp, tol=tol, check=check)


def nn(e, a="nA", b="nB"):
    return (e[a], e[b])


def asd_it(i, name, label, where, kind="or", dp=4, check=False):
    return it(i, "asd", kind, label, where, f"ppi_estimates.json per_db.e.analyses.{name}", lambda S, m=name: _or(S["c2"][m]),
              lambda r, m=name: r[m]["OR"], ("lor", name), lambda r, m=name: nn(r[m]), dp=dp, check=check)


def r7_it(i, name, label, where, kind, check=False):
    return it(i, "r7", kind, label, where, f"r7.json per_key.c2e.{name}", lambda S, m=name: _or(S["r7"]["per_key"]["c2e"][m]),
              lambda r, m=name: r[m]["OR"], ("lor", name), lambda r, m=name: nn(r[m]), check=check)


def r11_it(i, name, label, where, kind, check=False):
    k = "HR" if kind == "hr" else "OR"
    return it(i, "r11", kind, label, where, f"r11.json db.e.{name}", lambda S, m=name, k=k: _lohi(S["r11"][m], k),
              lambda r, m=name, k=k: r[m][k], ("lor", name),
              (lambda r, m=name: nn(r[m], "n_inf_A", "n_inf_B")) if kind == "hr" else (lambda r, m=name: nn(r[m])),
              tol=0.002 if kind == "hr" else None, check=check)


def r10_it(i, name, label, where, kind):
    return it(i, "r10", kind, label, where, f"r10.json per_key.c2e.{name}", lambda S, m=name: _or(S["r10"][m]),
              lambda r, m=name: r[m]["OR"], ("lor", name), lambda r, m=name: nn(r[m]))


def c1_it(i, name, label, where, kind, scale="OR"):
    return it(i, "c1", kind, label, where, f"r16.json famdose.e.est.{name}" + ("" if scale == "OR" else f" ({scale})"),
              lambda S, m=name, s=scale: _or(S["c1"][m], s), lambda r, m=name, s=scale: r[m][s], ("risk", name, scale), lambda r, m=name: nn(r[m]))


def e1_or(i, stratum, level, label, where, check=False):
    return it(i, "e1", "or", label, where, f"ppi_explore_E1.json db.e.strata.{stratum}.{level}", lambda S, s=stratum, v=level: _ci(S["e1"]["e"]["strata"][s][v]),
              lambda r, s=stratum, v=level: r["strata"][s][v]["OR"], ("lor", f"{stratum}:{level}"), lambda r, s=stratum, v=level: nn(r["strata"][s][v]), check=check)


def e1_ratio(i, stratum, level, ref, label, where):
    return it(i, "e1", "ratio", label, where, f"ppi_explore_E1.json db.e.strata.{stratum}.{level}.ratio_vs_ref",
              lambda S, s=stratum, v=level: (S["e1"]["e"]["strata"][s][v]["ratio_vs_ref"],) + tuple(S["e1"]["e"]["strata"][s][v]["ratio_CI"]),
              lambda r, s=stratum, v=level, f=ref: r["strata"][s][v]["OR"] / r["strata"][s][f]["OR"], ("lor", f"{stratum}:{level}:ratio"))


def e1_std(i, name, label, where):
    return it(i, "e1", "or", label, where, f"ppi_explore_E1.json db.e.{name}", lambda S, m=name: _ci(S["e1"]["e"][m]),
              lambda r, m=name: r[m], ("lor", name))


def r14_it(i, name, label, where, kind="or", check=False):
    k = "HR" if kind == "hr" else "OR"
    return it(i, "r14", kind, label, where, f"r14.json est.{name}", lambda S, m=name, k=k: _lohi(S["r14"]["est"][m], k),
              lambda r, m=name, k=k: r[m][k], ("lor", name), (lambda r, m=name: nn(r[m])) if kind != "hr" else None,
              tol=0.002 if kind == "hr" else None, check=check)


def r14_ratio(i, key, num, den, label, where):
    return it(i, "r14", "ratio", label, where, f"r14.json ratio.{key}", lambda S, m=key: _lohi(S["r14"]["ratio"][m], "ratio"),
              lambda r, a=num, b=den: r[a]["OR"] / r[b]["OR"], ("ratio", num, den))


ITEMS = [
    asd_it("B1-01", "hyp", "Censored at ICU discharge (inverse probability of censoring weighting)", "Table S3; main text, Results"),
    asd_it("B1-02", "pp", "Per protocol", "Table S3; main text, Methods"),
    asd_it("B1-03", "iptw", "Inverse probability of treatment weighting", "Table S3; main text, Methods"),
    asd_it("B1-04", "cc", "Complete cases", "Table S3; main text, Results"),
    asd_it("B1-05", "crude", "Unweighted (descriptive)", "Tables S3 and S4; main text, Results", dp=2),
    asd_it("B1-06", "death", "In-hospital death", "Table S3; main text, Results"),
    r7_it("B1-07", "m1_primary", "Additional adjustment: diabetes and insulin (M1)", "Table S3; main text, Methods", "or"),
    r7_it("B1-08", "m2_primary", "Additional adjustment: plus current-admission codes (M2)", "Table S3; main text, Methods", "or"),
    r7_it("B1-09", "m2_rass3", "M2, RASS ≥ −3 at time zero", "Table S3", "or"),
    r11_it("B1-10", "m3_primary", "Additional adjustment: corticosteroids and types of sedative (M3)", "Table S3; main text, Methods", "or"),
    r10_it("B1-11", "mlps_primary", "Gradient-boosted propensity score", "Table S3; main text, Methods", "or"),
    r10_it("B1-12", "dr_primary", "Doubly robust (gradient-boosted models)", "Table S3; main text, Methods", "or"),
    r11_it("B1-13", "ic6", "Interval-censored hazard ratio, baseline in 6 pieces", "Table S3", "hr"),
    r7_it("B1-14", "m1_negctl", "Negative control, M1", "Table S3", "nc"),
    r7_it("B1-15", "m2_negctl", "Negative control, M2", "Table S3", "nc"),
    r11_it("B1-16", "m3_negctl", "Negative control, M3", "Table S3", "nc"),
    r10_it("B1-17", "mlps_negctl", "Negative control, gradient-boosted propensity score", "Table S3", "nc"),
    r10_it("B1-18", "dr_negctl", "Negative control, doubly robust", "Table S3", "nc"),
    r7_it("B1-19", "px_cam", "Delirium in the subset of the antipsychotic outcome", "Table S6", "or"),
    r7_it("B1-20", "px_icu", "First antipsychotic in the ICU", "Tables S6 and S23, Figure 5, Abstract, main text, Results and Discussion", "ap"),
    it("B1-21", "r7", "ratio", "Antipsychotic ÷ delirium (same patients)", "Table S6 note; main text, Results and Discussion", "r7.json ratio.c2e",
       lambda S: (S["r7"]["per_key"]["c2e"]["px_icu"]["OR"] / S["r7"]["per_key"]["c2e"]["px_cam"]["OR"], S["r7"]["ratio"]["c2e"]["lo"], S["r7"]["ratio"]["c2e"]["hi"]),
       lambda r: r["px_icu"]["OR"] / r["px_cam"]["OR"], ("ratio", "px_icu", "px_cam"), tol=5e-4),
    it("B1-22", "a16", "ap", "First antipsychotic in the ICU without the exclusion for a positive screen before the first dose", "Table S23; main text, Results and Discussion", "r16_a16.json db.e.nodel.est.px_icu",
       lambda S: _or(S["a16"]["px_icu"]), lambda r: r["px_icu"]["OR"], ("risk", "px_icu", "OR"), lambda r: nn(r["px_icu"])),
    c1_it("B1-23", "primary", "Dose comparison (≤20 vs ≥40 mg/day), odds ratio", "Tables S9, S13 and S18, Figure 3b", "or_dose"),
    c1_it("B1-24", "primary", "Dose comparison, risk ratio", "Table S13", "rr", "RR"),
    c1_it("B1-25", "primary", "Dose comparison, risk difference", "Table S13", "rd", "RD"),
    c1_it("B1-26", "kadj_primary", "Dose comparison, chronic kidney disease and kidney replacement therapy added", "Table S18; main text, Results", "or_dose"),
    c1_it("B1-27", "kexc_primary", "Dose comparison, patients on kidney replacement therapy removed", "Table S18; main text, Results", "or_dose"),
    c1_it("B1-28", "bleedexc_primary", "Dose comparison, exclusions for bleeding and PPI infusion", "Table S18; main text, Results", "or_dose"),
    e1_or("B1-29", "vent", "no", "No invasive ventilation at time zero (APACHE flag)", "Tables S4 and S11 E, Figure 4, Discussion"),
    e1_or("B1-30", "vent", "yes", "Invasive ventilation at time zero (APACHE flag)", "Tables S4 and S11 E, Figure 4, Discussion"),
    e1_ratio("B1-31", "vent", "yes", "no", "Ventilated ÷ not ventilated", "Tables S4 and S11 E, Figure 4, main text, Results and Discussion"),
    e1_ratio("B1-32", "rass", "deep", "light", "Deep sedation ÷ awake or light sedation (run of E1)", "Table S4, Figure 4, main text, Results"),
    e1_std("B1-33", "std_own", "standardised to its own RASS mix", "Table S4 note; main text, Results"),
    e1_std("B1-34", "std_other", "standardised to the RASS mix of MIMIC-IV", "Table S4 note; main text, Results and Discussion"),
    r14_it("B1-35", "pre_primary", "Model with information before time zero only", "Table S11 D, Discussion"),
    r14_ratio("B1-36", "primary_pre_vs_main", "pre_primary", "main_primary", "Pre-time-zero model ÷ main model", "Table S11 D, Discussion"),
    r14_it("B1-37", "strat_pre_0", "Not ventilated by time-stamped records (pre-time-zero model)", "Table S11 E"),
    r14_it("B1-38", "strat_pre_1", "Ventilated by time-stamped records (pre-time-zero model)", "Table S11 E"),
    r14_ratio("B1-39", "strat_pre", "strat_pre_1", "strat_pre_0", "By time-stamped records: ventilated ÷ not ventilated", "Table S11 E, Figure 4 legend"),
]

CHECKS = [asd_it(f"c-asd-{n}", n, n, "", check=True) for n in ("primary", "del72", "negctl", "rass3", "egfr60", "age65")] + [
    r7_it("c-r7-m0p", "m0_primary", "m0_primary", "", "or", check=True), r7_it("c-r7-m0n", "m0_negctl", "m0_negctl", "", "or", check=True),
    r11_it("c-r11-p", "primary", "primary", "", "or", check=True), r11_it("c-r11-n", "negctl", "negctl", "", "or", check=True),
    r11_it("c-r11-ic", "ic", "ic", "", "hr", check=True),
    it("c-a16-rr", "a16", "rr", "px_icu RR", "", "", lambda S: _or(S["a16"]["px_icu"], "RR"), lambda r: r["px_icu"]["RR"], ("risk", "px_icu", "RR"), check=True),
    it("c-a16-rd", "a16", "rd", "px_icu RD", "", "", lambda S: _or(S["a16"]["px_icu"], "RD"), lambda r: r["px_icu"]["RD"], ("risk", "px_icu", "RD"), check=True),
    it("c-e1-p", "e1", "or", "primary", "", "", lambda S: _ci(S["e1"]["e"]["primary"]), lambda r: r["primary"]["OR"], ("lor", "primary"), check=True),
    r14_it("c-r14-p", "main_primary", "main_primary", "", check=True), r14_it("c-r14-ic", "main_ic", "main_ic", "", "hr", check=True),
]

PAT_REF = [("asd", ("lor", "primary"), "primary", None), ("asd", ("lor", "del72"), "del72", None), ("asd", ("lor", "negctl"), "negctl", None),
           ("asd", ("lor", "rass3"), "rass3_primary", None), ("asd", ("lor", "egfr60"), "egfr60", None), ("asd", ("lor", "age65"), "age65", None),
           ("r7", ("lor", "m0_primary"), "primary", None), ("r7", ("lor", "m0_negctl"), "negctl", None),
           ("r11", ("lor", "primary"), "primary", None), ("r11", ("lor", "negctl"), "negctl", None), ("r11", ("lor", "ic"), "ic:ic_all", 0.002),
           ("e1", ("lor", "primary"), "primary", 0.002), ("r14", ("lor", "main_primary"), "primary", None), ("r14", ("lor", "main_ic"), "ic:ic_all", 0.002)]


def bcol(B, name):
    return B["v"][:, B["cols"].index(name)]


def pct_log(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < MIN_BOOT:
        return None
    return float(np.exp(np.percentile(v, 2.5))), float(np.exp(np.percentile(v, 97.5))), int(v.size)


def interval(spec, B):
    if B is None:
        return None
    if spec[0] == "lor":
        return pct_log(bcol(B, spec[1]))
    if spec[0] == "ratio":
        return pct_log(bcol(B, spec[1]) - bcol(B, spec[2]))
    sca = r16.scales(bcol(B, spec[1] + ":A"), bcol(B, spec[1] + ":B"))[spec[2]]
    q = r16.pctl(sca)
    if q is None:
        return None
    lo, hi, _, k = q
    return (math.exp(lo), math.exp(hi), k) if spec[2] != "RD" else (lo, hi, k)


THRESH = {"or": (0.75, 1.0), "or_dose": (0.75, 1.0), "nc": (1.0,), "ap": (1.0,), "hr": (1.0,), "ratio": (1.0,), "rr": (), "rd": ()}
READ_ZH = {"favours": "association favouring PPI", "favours_dose": "association favouring ≤20 mg/day", "benefit_excluded": "clinically important benefit excluded",
           "opposite": "association in the opposite direction", "inconclusive": "inconclusive", "signal": "signal of residual confounding", "no_signal": "no signal of residual confounding",
           "fewer": "fewer in the PPI arm", "ap_opposite": "opposite direction", "no_direction": "inconclusive (direction unclear)", "association": "association (favouring PPI)",
           "hr_opposite": "opposite direction", "unclear": "association unclear", "higher": "numerator higher", "lower": "numerator lower", "no_difference": "no clear difference",
           "not_read": "not read separately", "no_interval": "no interval"}


def read(kind, lo, hi):
    if lo is None or hi is None:
        return "no_interval"
    if kind in ("or", "or_dose"):
        if hi < 1:
            return "favours_dose" if kind == "or_dose" else "favours"
        if lo > 1:
            return "opposite"
        return "benefit_excluded" if lo > asd.MCID else "inconclusive"
    if kind == "nc":
        return "signal" if (lo > 1 or hi < 1) else "no_signal"
    if kind == "ap":
        return "fewer" if hi < 1 else ("ap_opposite" if lo > 1 else "no_direction")
    if kind == "hr":
        return "association" if hi < 1 else ("hr_opposite" if lo > 1 else "unclear")
    if kind == "ratio":
        return "higher" if lo > 1 else ("lower" if hi < 1 else "no_difference")
    return "not_read"


def at_threshold(kind, lo, hi, dp=4):
    out = []
    for t in THRESH.get(kind, ()):
        for lab, x in (("lower", lo), ("upper", hi)):
            if x is not None and round(x, 2) == t:
                out.append(f"{lab} limit {x:.{dp}f} rounds to {t:.2f}")
    return out


def same(x, ref, dp=4, tol=None):
    if x is None or ref is None or not np.isfinite(x):
        return False
    return abs(x - ref) <= tol + 1e-12 if tol is not None else abs(round(float(x), dp) - float(ref)) < 1e-9


def configure(mimic, eicu):
    for p in (mimic, eicu, os.path.join(eicu, "eicu", "medication.parquet")):
        if not os.path.exists(p):
            raise SystemExit("database path not found")
    r7.SRC = {"m": mimic, "e": eicu}
    r14.E_ROOT = eicu


def extract(args):
    P = priv(args)
    configure(os.path.abspath(args.mimic), os.path.abspath(args.eicu))
    t0, start = time.time(), now_iso()
    ns = argparse.Namespace
    steps = [("E1 process and hospital intensity, eICU-CRD", lambda: e1.extract(ns(private=P, db="eicu", results=None)), ["e_c2_process", "e_hosp_intensity"]),
             ("E1 process, MIMIC-IV (RASS mix only)", lambda: e1.extract(ns(private=P, db="mimic", results=None)), ["m_c2_process"]),
             ("R7 covariates", lambda: r7.extra(ns(private=P, db="e", part="cov", results=None)), ["e_r7cov"]),
             ("R7 antipsychotic outcome", lambda: r7.extra(ns(private=P, db="e", part="out", results=None)), ["e_r7out"]),
             ("R11 assessment timeline", lambda: r11.extract(ns(private=P, db="e", part="timeline", results=None)), ["e_r11tl"]),
             ("R11 covariates (M3)", lambda: r11.extract(ns(private=P, db="e", part="cov", results=None)), ["e_r11cov"]),
             ("R14 ventilation and lactate before time zero", lambda: r14.extract(ns(private=P, results=None)), ["e_r14pre"])]
    for label, fn, files in steps:
        if all(os.path.exists(os.path.join(P, f + ".parquet")) for f in files):
            say(f"{label}: already present, skipped")
            continue
        fn()
        say(f"{label}: written")
    record(P, "extract", t0, start)


def estimate(args):
    P, R = priv(args), res_dir(args)
    t0, start = time.time(), now_iso()
    S = load_stored(R)
    out = {}
    for p in ORDER:
        lines, good = [], True
        try:
            st = PIPES[p]["load"](P)
            res, cache = PIPES[p]["point"](st)
        except Exception as e:
            out[p] = {"good": False, "lines": [f"FAIL | {p}: could not be computed ({type(e).__name__})"]}
            say(out[p]["lines"][0])
            continue
        rj = asd.to_json(res)
        for x in [i for i in ITEMS + CHECKS if i["pipe"] == p]:
            ref = x["stored"](S)[0]
            v = x["point"](rj)
            ok = same(v, ref, x["dp"], x["tol"])
            good &= ok
            lines.append(f"{'PASS' if ok else 'FAIL'} | {p} {x['id']} point estimate: {v if v is None else round(v, x['dp'])} vs {ref}")
        if p == "e1":
            for lv, ref in S["e1"]["m"]["rass_shares"].items():
                ok = same(st["shares_m"][lv], ref)
                good &= ok
                lines.append(f"{'PASS' if ok else 'FAIL'} | e1 MIMIC-IV RASS mix {lv}: {st['shares_m'][lv]:.4f} vs {ref}")
        est = {"res": rj, "flow": {k: sc.mask_n(v) for k, v in (st.get("flow") or {}).items()}, "n": int(len(st["d"]["A"]))}
        if p == "e1":
            est["shares_m"] = st["shares_m"]
        json.dump(est, open(pr(P, f"est_{p}.json"), "w", encoding="utf-8"))
        np.savez(pr(P, f"cache_{p}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        out[p] = {"good": bool(good), "lines": lines}
        for ln in lines:
            print(ln, flush=True)
        say(f"{p}: point estimates {'reproduced' if good else 'NOT reproduced'}")
    json.dump(out, open(pr(P, "point_checks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    record(P, "estimate", t0, start)
    if not out.get("asd", {}).get("good"):
        raise SystemExit("asd.py PPI comparison eICU-CRD not reproduced from the private folder: stop (addendum 3.1)")


def load_state(P, p):
    est = json.load(open(pr(P, f"est_{p}.json"), encoding="utf-8"))
    st = PIPES[p]["load"](P, est)
    z = np.load(pr(P, f"cache_{p}.npz"))
    return st, {k.replace("__", ":"): z[k] for k in z.files}


def gate_ok(P, p):
    f = pr(P, "gate.json")
    return os.path.exists(f) and json.load(open(f, encoding="utf-8")).get(p, {}).get("status") == "PASS"


def boot_path(P, p, scheme, a, b):
    return pr(P, f"boot_{p}_{scheme}_{a:04d}_{b:04d}.npz")


def boot(args):
    P = priv(args)
    p, scheme = args.pipe, args.scheme
    if scheme == "pat" and not gate_ok(P, p):
        raise SystemExit(f"{p}: the reproduction gate has not been passed; no patient-level replicates")
    t0 = time.time()
    st, cache = load_state(P, p)
    cols, nb = PIPES[p]["cols"], PIPES[p]["nboot"]
    reps = list(range(args.start, min(args.stop, nb)))
    V = np.full((len(reps), len(cols)), np.nan)
    failed = 0
    for i, rep in enumerate(reps):
        try:
            V[i] = np.asarray(PIPES[p]["vec"](st, scheme, rep, cache), float)
        except Exception:
            failed += 1
    np.savez(boot_path(P, p, scheme, args.start, args.stop), reps=np.array(reps), v=V, cols=np.array(cols), failed=failed)
    say(f"{p} {scheme} {args.start}-{args.stop} done, failed {failed} ({time.time() - t0:.0f} s)")


def load_boots(P, p, scheme):
    files = sorted(glob.glob(pr(P, f"boot_{p}_{scheme}_*.npz")))
    if not files:
        return None
    reps, vs, cols, failed = [], [], None, 0
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        vs.append(z["v"])
        cols = [str(x) for x in z["cols"]]
        failed += int(z["failed"])
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"cols": cols, "v": np.concatenate(vs)[first], "n": int(first.size), "failed": failed}


def run(args):
    P = priv(args)
    scheme = args.scheme
    pc = json.load(open(pr(P, "point_checks.json"), encoding="utf-8"))
    pipes = [p for p in ORDER if pc.get(p, {}).get("good") and (scheme == "hosp" or gate_ok(P, p))]
    jobs = []
    for p in pipes:
        nb, step = PIPES[p]["nboot"], PIPES[p]["chunk"]
        for a in range(0, nb, step):
            b = min(a + step, nb)
            if not os.path.exists(boot_path(P, p, scheme, a, b)):
                jobs.append((p, a, b))
    t0, start = time.time(), now_iso()
    say(f"run {scheme}: {len(jobs)} chunks of {', '.join(pipes)} on {args.workers} processes")
    running, done = [], []
    while jobs or running:
        while jobs and len(running) < args.workers:
            p, a, b = jobs.pop(0)
            lf = open(pr(P, "logs", f"{p}_{scheme}_{a:04d}.log"), "w", encoding="utf-8")
            cmd = [sys.executable, "-B", os.path.abspath(__file__), "boot", "--pipe", p, "--scheme", scheme, "--start", str(a), "--stop", str(b), "--private", P]
            running.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE), lf, p, a, b, time.time()))
        time.sleep(2)
        for job in list(running):
            proc, lf, p, a, b, ts = job
            if proc.poll() is not None:
                lf.close()
                running.remove(job)
                done.append({"pipe": p, "start": a, "stop": b, "rc": proc.returncode, "seconds": round(time.time() - ts, 1)})
                if proc.returncode != 0:
                    say(f"chunk {p} {scheme} {a}-{b} ended with code {proc.returncode}")
                elif len(done) % 10 == 0:
                    say(f"{len(done)} chunks done, {len(jobs)} waiting, {(time.time() - t0) / 60:.1f} min")
    bad = [x for x in done if x["rc"] != 0]
    record(P, f"run_{scheme}", t0, start, {"workers": args.workers, "chunks": len(done), "failed_chunks": len(bad),
                                           "cpu_seconds": round(sum(x["seconds"] for x in done), 1), "pipes": pipes})
    say(f"run {scheme} finished: {len(done)} chunks, {len(bad)} with errors, {(time.time() - t0) / 60:.1f} min")
    if bad:
        raise SystemExit(1)


def gate(args):
    P, R = priv(args), res_dir(args)
    t0, start = time.time(), now_iso()
    S = load_stored(R)
    pc = json.load(open(pr(P, "point_checks.json"), encoding="utf-8"))
    out = {}
    for p in ORDER:
        lines = list(pc.get(p, {}).get("lines", []))
        good = bool(pc.get(p, {}).get("good"))
        H = load_boots(P, p, "hosp")
        if H is None or H["n"] < PIPES[p]["nboot"]:
            good = False
            lines.append(f"FAIL | {p}: hospital-level replicates missing or incomplete")
        else:
            for x in [i for i in ITEMS + CHECKS if i["pipe"] == p]:
                _, slo, shi = x["stored"](S)
                q = interval(x["spec"], H)
                tol = x["tol"] if x["kind"] == "hr" else None
                ok = q is not None and same(q[0], slo, x["dp"], tol) and same(q[1], shi, x["dp"], tol)
                good &= ok
                nd = x["dp"]
                lines.append(f"{'PASS' if ok else 'FAIL'} | {p} {x['id']} hospital-level interval: "
                             + (f"{q[0]:.{nd}f}-{q[1]:.{nd}f}" if q else "none") + f" vs {slo}-{shi}")
        out[p] = {"status": "PASS" if good else "FAIL", "lines": lines, "hosp_failed_reps": (H or {}).get("failed")}
        for ln in lines:
            if ln.startswith("FAIL") or "hospital-level" in ln:
                print(ln, flush=True)
        say(f"gate {p}: {out[p]['status']}")
    json.dump(out, open(pr(P, "gate.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    record(P, "gate", t0, start)


BATCH2 = [
    ("B2-01", "PPI comparison instrumental-variable risk difference (Table S3 note; not read)", "asd.py; ppi_estimates.json per_db.e.iv", "−0.071 (−0.168 to 0.212)"),
    ("B2-02", "First quetiapine in the ICU (Table S6)", "r7.py; r7.json per_key.c2e.px_icu_oq", "0.89 (0.65–1.19)"),
    ("B2-03–05", "Ratios of screens per ICU day before the first positive, of assessable screens and of 12-h periods with an assessable screen (Table S2; the second also in the Table S15 note)",
     "r11.py; r11.json db.e.screen", "1.14 (0.97–1.28), 1.13 (0.96–1.29), 1.10 (0.95–1.22)"),
    ("B2-06–08", "Negative control of the dose comparison: main analysis, kidney covariates added, patients on kidney replacement therapy removed (Tables S9 and S18)", "r16.py; r16.json famdose.e.est",
     "1.11 (0.78–1.82), 1.12 (0.77–1.80), 1.10 (0.65–1.71)"),
    ("B2-09–10", "Delirium assessment in the 24 h before time zero; none ÷ some (Table S4)", "explore_ppi.py; strata.base", "0.85 (0.71–0.98); 1.19 (1.00–1.94)"),
    ("B2-11–13", "ICU admission not from and from the operating room, and their ratio (Table S4)", "explore_ppi.py; strata.surg", "0.94 (0.84–1.07), 1.37 (0.82–3.14); 1.46 (0.87–3.48)"),
    ("B2-14–18", "Hospital screening intensity high, middle, low; middle ÷ high, low ÷ high (Table S4)", "explore_ppi.py; strata.hosp",
     "1.07 (0.79–1.58), 0.94 (0.76–1.33), 1.00 (0.80–1.25); 0.88 (0.54–1.36), 0.93 (0.58–1.36)"),
    ("B2-19–20", "Moderate ÷ awake or light, not recorded ÷ awake or light (run of E1, Table S4)", "explore_ppi.py; strata.rass", "1.84 (1.38–3.03), 2.13 (1.32–3.72)"),
    ("B2-21–22", "Odds ratio of ICU discharge alive within 168 h; ratio of detection per assessment (Table S2 note)", "explore_ppi.py; discharge, per_assessment", "0.87 (0.70–1.03); 0.84 (0.74–1.01)"),
    ("B2-23–26", "Interval-censored hazard ratio of the pre-time-zero model and its ratio; negative control and its ratio (Table S11 D)", "r14.py; r14.json",
     "0.91 (0.83–1.06), 1.01 (0.99–1.02); 0.99 (0.86–1.22), 0.99 (0.98–1.02)"),
    ("B2-27–36", "Dose comparison: 72 h, censored at ICU discharge, per protocol, inverse probability of treatment weighting, complete cases, RASS ≥ −3, eGFR <50, unweighted, in-hospital death, instrumental variable (Table S9 and note)",
     "asd.py; famdose_estimates.json per_db.e", "0.89 (0.51–1.63), 1.32 (0.58–3.83), 0.76 (0.47–1.23), 0.85 (0.49–1.47), 1.24 (0.60–2.59), "
     "1.06 (0.50–2.02), 0.94 (0.51–1.68), 0.81 (0.56–1.37), 1.12 (0.86–1.59); −0.003 (−0.354 to 0.399)"),
    ("B2-37–42", "Dose comparison: M1, M2 and their negative controls; the other classification of eICU order doses and its negative control (Table S9)", "r7.py; r7.json per_key.c1e, dose",
     "0.94 (0.55–1.72), 0.93 (0.55–1.69), 1.16 (0.83–2.16), 1.16 (0.84–2.25), 0.95 (0.58–1.65), 1.20 (0.86–1.97)"),
    ("B2-43–45", "Burden outcomes of the dose comparison (differences in days, Table S9)", "e2_burden.py; e2_burden.json famdose.e", "−0.05 (−0.29 to 0.20), 0.02 (−0.13 to 0.22), 0.00 (−0.30 to 0.41)"),
    ("B2-46–54", "Moderate, deep and not recorded, each against awake or light, on three scales (run of R16, Table S14)", "r16.py; r16.json interaction.e (the patient-level replicates of R16 can be summarised directly)",
     "odds ratio 1.84 (1.38–3.03), 1.91 (1.07–3.09), 2.13 (1.32–3.72); risk ratio 1.64 (1.30–2.46), 1.63 (1.10–2.29), 2.00 (1.29–3.38); "
     "risk difference 9.1 (3.8 to 18.3), 10.8 (−1.9 to 21.0), 6.0 (2.3 to 9.9)"),
    ("B2-55–59", "Ratio of assessable screens per ICU day: RASS ≥ −3, awake or light, moderate, deep, not recorded (Table S15 note)", "r16.py; r16.json db.e.scr (as above)",
     "1.12 (1.00–1.23), 1.09 (0.97–1.17), 1.40 (0.98–1.83), 1.01 (0.33–2.21), 1.16 (0.91–1.57)"),
]


def num(kind, v, dp=2):
    if v is None:
        return "—"
    s = f"{100 * v:.1f}" if kind == "rd" else f"{v:.{dp}f}"
    return s.replace("-", "−")


def span(kind, lo, hi, dp=2):
    if lo is None or hi is None:
        return "no interval"
    return f"{num(kind, lo, dp)}{' to ' if kind == 'rd' else '–'}{num(kind, hi, dp)}"


def report(args):
    P, R = priv(args), res_dir(args)
    fj, fm = os.path.join(R, "r18.json"), os.path.join(R, "r18.md")
    if (os.path.exists(fj) or os.path.exists(fm)) and not args.overwrite:
        raise SystemExit("results/r18.json or r18.md exists (use --overwrite)")
    t0, start = time.time(), now_iso()
    S = load_stored(R)
    gt = json.load(open(pr(P, "gate.json"), encoding="utf-8"))
    est = {p: json.load(open(pr(P, f"est_{p}.json"), encoding="utf-8")) for p in ORDER if os.path.exists(pr(P, f"est_{p}.json"))}
    BH = {p: load_boots(P, p, "hosp") for p in ORDER}
    BP = {p: load_boots(P, p, "pat") if gt.get(p, {}).get("status") == "PASS" else None for p in ORDER}
    checks, good5 = [], True
    for p, spec, ref, tol in PAT_REF:
        if BP.get(p) is None:
            continue
        q = interval(spec, BP[p])
        src = S["icpat"][ref.split(":")[1]] if ref.startswith("ic:") else S["pat"][ref]
        rlo, rhi = (src["lo"], src["hi"]) if ref.startswith("ic:") else (src["OR_lo"], src["OR_hi"])
        ok = q is not None and same(q[0], rlo, 4, tol) and same(q[1], rhi, 4, tol)
        good5 &= ok
        checks.append(f"{'PASS' if ok else 'FAIL'} | {p} {spec[1]}, patient-level interval vs R16 ({ref}): "
                      + (f"{q[0]:.4f}-{q[1]:.4f}" if q else "none") + f" vs {rlo}-{rhi}")
    for ln in checks:
        print(ln, flush=True)
    if not good5:
        raise SystemExit("the patient-level draws do not reproduce R16's patient-level intervals: nothing written (addendum 3.5)")
    rows = []
    for x in ITEMS:
        p = x["pipe"]
        sv = x["stored"](S)
        r = (est.get(p) or {}).get("res")
        nv = x["n"](r) if (x["n"] and r) else None
        qh, qp = interval(x["spec"], BH.get(p)), interval(x["spec"], BP.get(p))
        dp = x["dp"]
        rnd = (lambda v: None if v is None else round(float(v), dp))
        row = {"id": x["id"], "label": x["label"], "where": x["where"], "script": x["script"], "key": x["key"], "type": x["kind"],
               "n": {"PPI_or_low_dose": sc.mask_n(nv[0]), "famotidine_or_high_dose": sc.mask_n(nv[1])} if nv else None,
               "estimate": rnd(x["point"](r)) if r else rnd(sv[0]), "estimate_reported": sv[0],
               "hospital": {"lo": sv[1], "hi": sv[2], "rerun_lo": rnd(qh[0]) if qh else None, "rerun_hi": rnd(qh[1]) if qh else None,
                            "valid": qh[2] if qh else None}}
        pv = x["point"](r) if r else sv[0]
        hl, hh = (qh[0], qh[1]) if qh else (sv[1], sv[2])
        pl, ph = (qp[0], qp[1]) if qp else (None, None)
        row["display"] = {"estimate": num(x["kind"], pv), "hospital": span(x["kind"], hl, hh), "patient": span(x["kind"], pl, ph)}
        if gt.get(p, {}).get("status") != "PASS":
            row["patient"] = None
            row["note"] = f"{PIPES[p]['script']} did not pass the reproduction gate: no patient-level interval (addendum 3.4)"
            rows.append(row)
            continue
        rh, rp = read(x["kind"], hl, hh), read(x["kind"], pl, ph)
        row["hospital"].update({"reading": rh, "reading_zh": READ_ZH[rh], "at_threshold": at_threshold(x["kind"], hl, hh, dp)})
        row["patient"] = {"lo": rnd(pl), "hi": rnd(ph), "valid": qp[2] if qp else None, "failed": BP[p]["failed"], "reading": rp,
                          "reading_zh": READ_ZH[rp], "at_threshold": at_threshold(x["kind"], pl, ph, dp)}
        row["agree"] = None if rh == "not_read" else bool(rh == rp)
        rows.append(row)
    T = json.load(open(pr(P, "timing.json"), encoding="utf-8")) if os.path.exists(pr(P, "timing.json")) else {"steps": []}
    T["steps"].append({"step": "report", "start": start, "end": now_iso(), "seconds": round(time.time() - t0, 1)})
    out = {"plan": PLAN, "software": {"python": sys.version.split()[0], "numpy": np.__version__, **versions()},
           "private_folder": "the private folder of R16 (outside the repository), outputs in its r18 subfolder",
           "gate": {p: gt.get(p, {}).get("status") for p in ORDER},
           "checks": {"point_and_hospital": {p: gt.get(p, {}).get("lines", []) for p in ORDER}, "patient_vs_R16": checks},
           "boot": {p: {"hosp": {"reps": (BH[p] or {}).get("n"), "failed": (BH[p] or {}).get("failed")},
                        "pat": {"reps": (BP[p] or {}).get("n"), "failed": (BP[p] or {}).get("failed")}} for p in ORDER},
           "batch1": rows, "differ": [r["id"] for r in rows if r.get("agree") is False],
           "batch2_remaining": [{"ids": a, "estimates": b, "source": c, "reported": d} for a, b, c, d in BATCH2],
           "timing": T}
    json.dump(asd.to_json(out), open(fj, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    open(fm, "w", encoding="utf-8", newline="\n").write(md(out))
    say(f"results/r18.json and r18.md written; {len(out['differ'])} estimates whose reading depends on the resampling unit")


def versions():
    import duckdb
    import pandas
    import scipy
    import sklearn
    return {"scipy": scipy.__version__, "duckdb": duckdb.__version__, "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}


def md(o):
    L = ["# Revision R18 results: patient-level bootstrap intervals of the eICU-CRD estimates (first batch)", "",
         f"Plan: {o['plan']}. Each estimate is computed by the function of the script that first wrote it; only the resampling changes from hospitals to patients"
         " (2,000 replicates, seeds 20261013 plus the replicate number, as in R12 and R16; 1,000 for the flexible models of R10); the propensity score is refitted in every replicate, and the hospital indicators are kept as in the main model."
         "The hospital-level column holds the intervals reported in Additional file 1, rerun and reproduced here with the original code and seeds. Odds ratios, risk ratios and ratios <1 mean lower in the PPI arm (in the dose comparison, the ≤20 mg/day arm)."
         "Counts masked: 1–9 as <10, otherwise the nearest 5. Readings follow addendum section 4, with unrounded limits; at the threshold means a limit equal to 0.75 or 1.00 when rounded to two decimals.", "",
         f"Environment: Python {o['software']['python']}, NumPy {o['software']['numpy']}, SciPy {o['software']['scipy']}, DuckDB {o['software']['duckdb']}, "
         f"pandas {o['software']['pandas']}, scikit-learn {o['software']['scikit-learn']}.", "",
         "## 1. Reproduction gate", "", "| Pipeline | Reproduction of the point estimates and hospital-level intervals | Hospital-level resampling (replicates, failed) | Patient-level resampling (replicates, failed) |", "|---|---|---|---|"]
    for p in ORDER:
        b = o["boot"][p]
        L.append(f"| {PIPES[p]['script']} | {'passed' if o['gate'].get(p) == 'PASS' else 'not passed'} | {b['hosp']['reps']}, {b['hosp']['failed']} | "
                 f"{b['pat']['reps'] if b['pat']['reps'] is not None else 'not computed'}, {b['pat']['failed'] if b['pat']['failed'] is not None else '—'} |")
    nfail = sum(1 for p in ORDER for ln in o["checks"]["point_and_hospital"][p] if ln.startswith("FAIL"))
    npass = sum(1 for p in ORDER for ln in o["checks"]["point_and_hospital"][p] if ln.startswith("PASS"))
    L += ["", f"Item checks: {npass} passed, {nfail} failed (each in r18.json checks)."]
    for p in ORDER:
        for ln in o["checks"]["point_and_hospital"][p]:
            if ln.startswith("FAIL"):
                L.append(f"- {ln}")
    L += ["", "Check of the patient-level draws against R12 and R16 (with the same patient-level draws, estimates that already had patient-level intervals should get the same intervals): ", ""]
    L += [f"- {ln}" for ln in o["checks"]["patient_vs_R16"]]
    L += ["", "## 2. First batch, 39 estimates", "",
          "| ID | Estimate | Tables, figures, main text | Patients PPI / famotidine | Estimate (hospital-level 95% CI, as reported) | Patient-level 95% CI (usable replicates) | Reading: hospital → patient level | Same |",
          "|---|---|---|---|---|---|---|---|"]
    for r in o["batch1"]:
        n = f"{r['n']['PPI_or_low_dose']} / {r['n']['famotidine_or_high_dose']}" if r.get("n") else "—"
        h = f"{r['display']['estimate']} ({r['display']['hospital']})"
        if r.get("patient") is None:
            L.append(f"| {r['id']} | {r['label']} | {r['where']} | {n} | {h} | not computed ({r.get('note', '')}) | — | — |")
            continue
        pt = r["patient"]
        pc = r["display"]["patient"]
        flag = ""
        if r["hospital"]["at_threshold"] or pt["at_threshold"]:
            flag = "; at the threshold: " + ", ".join(([f"hospital level {s}" for s in r["hospital"]["at_threshold"]]) + [f"patient level {s}" for s in pt["at_threshold"]])
        agree = "—" if r["agree"] is None else ("same" if r["agree"] else "**differ**")
        L.append(f"| {r['id']} | {r['label']} | {r['where']} | {n} | {h} | {pc} ({pt['valid']}) | {r['hospital']['reading_zh']} → {pt['reading_zh']}{flag} | {agree} |")
    L += ["", "Two-decimal values are rounded once from the unrounded numbers (as in r16_fmt.py). The risk difference (B1-25) is in percentage points. B1-05 (unweighted) is reported to two decimals only (as in asd.py)."
          "The point estimate of B1-21 is a ratio of two odds ratios; its hospital-level interval is taken from r7.json.", "",
          "## 3. Estimates whose reading depends on the resampling unit", ""]
    dif = [r for r in o["batch1"] if r.get("agree") is False]
    if not dif:
        L.append("None: every estimate of the first batch has the same reading with hospital-level and patient-level intervals.")
    else:
        L += ["| ID | Estimate | Hospital-level 95% CI | Patient-level 95% CI | Hospital-level reading | Patient-level reading |", "|---|---|---|---|---|---|"]
        for r in dif:
            nd = 2 if r["id"] == "B1-05" else 4
            f4 = lambda v: f"{v:.{nd}f}" if r["type"] != "rd" else f"{100 * v:.2f}"
            L.append(f"| {r['id']} | {r['label']} ({r['where']}) | {f4(r['hospital']['lo'])}–{f4(r['hospital']['hi'])} | "
                     f"{f4(r['patient']['lo'])}–{f4(r['patient']['hi'])} | {r['hospital']['reading_zh']} | {r['patient']['reading_zh']} |")
        L += ["", "As the Methods state, conclusions on eICU-CRD are kept only where the two intervals lead to the same reading; this file reports them only, and any change to the manuscript is decided separately."]
    L += ["", "## 4. Running time", "", "| Step | Start | End | Minutes | Note |", "|---|---|---|---|---|"]
    for s in o["timing"]["steps"]:
        extra = f"{s.get('workers', '')} processes, {s.get('chunks', '')} chunks, CPU {s['cpu_seconds'] / 3600:.1f} hours" if "cpu_seconds" in s else ""
        L.append(f"| {s['step']} | {s['start']} | {s['end']} | {s['seconds'] / 60:.1f} | {extra} |")
    L += ["", "## 5. Second batch (not computed; to be decided)", "",
          "Under addendum section 5, the list of 98 estimates is far longer than about 40, so the first batch, the estimates interpreted in the main text, was computed first; the 59 estimates below use the same method, gate and readings, with their code to be written before they are computed.", "",
          "| ID | Estimate (table) | Script and result key | As reported (hospital-level interval) |", "|---|---|---|---|"]
    for b in o["batch2_remaining"]:
        L.append(f"| {b['ids']} | {b['estimates']} | {b['source']} | {b['reported']} |")
    L += ["", "Estimates that already had patient-level intervals and are not recomputed, and numbers outside the scope (pooled estimates, ratios between the databases, P values, E-values, descriptive numbers), are listed in addendum sections 1.3 and 1.4.", ""]
    return "\n".join(L)


def synth_states(d, rng):
    n = len(d["A"])
    st = {"asd": {"d": d}}
    d0 = dict(d)
    d0["t_ap_icu"] = np.where(rng.random(n) < 0.10, rng.uniform(0, 160, n), np.nan)
    d0["t_oq_icu"] = np.where(rng.random(n) < 0.05, rng.uniform(0, 160, n), np.nan)
    d0["n_cam"] = rng.integers(0, 8, n).astype(float)
    d0["ap_pre"], d0["psych"] = rng.random(n) < 0.05, rng.random(n) < 0.03
    ex = np.column_stack([rng.random(n) < 0.25, rng.random(n) < 0.06, rng.random(n) < 0.12]).astype(float)
    X1, n1, _ = asd.prevalence_filter(np.column_stack([d["X"], ex]), d["names"] + ["diabetes", "insulin_inf_t0", "insulin_24h"])
    d1 = dict(d0, X=X1, names=n1)
    st["r7"] = {"dd": {"m0": d0, "m1": d1, "m2": dict(d1)}, "d": d0}
    nc = np.column_stack([rng.random(n) < q for q in (0.12, 0.06, 0.05, 0.04)]).astype(float)
    X3, n3, dr3 = asd.prevalence_filter(np.column_stack([d["X"], nc]), d["names"] + list(r11.NEWCOV))
    st["r11"] = {"d": dict(d0, X3=X3, names3=n3, dropped3=dr3)}
    st["r10"] = {"d": dict(d, extras={"age_years": d["extras"]["age_years"], "egfr": d["egfr"], "h_icu": d["h_icu"], "apache": rng.uniform(20, 120, n)},
                           rass=np.where(d["rgrp"] == "missing", np.nan, rng.integers(-5, 2, n).astype(float)))}
    st["a16"] = {"d": d0}
    st["c1"] = {"d": dict(d, bleed=rng.random(n) < 0.03)}
    lc = e1.LEVEL_CODE
    st["e1"] = {"d": dict(d, s_rass=np.array([lc["rass"][x] for x in d["rgrp"]]), s_base=rng.integers(0, 2, n), s_surg=rng.integers(0, 2, n),
                          s_vent=rng.integers(0, 2, n), s_hosp=rng.integers(0, 3, n), p_n_ok_until_pos=rng.integers(0, 12, n).astype(float)),
                "shares_m": {"light": 0.57, "moderate": 0.12, "deep": 0.23, "missing": 0.08}}
    dr = dict(d)
    r14.add_pre(dr, rng.random(n) < 0.3, rng.random(n) < 0.3, np.where(rng.random(n) < 0.5, np.nan, rng.uniform(0.5, 6, n)))
    st["r14"] = {"d": dr}
    return st


def selftest(args):
    sys.excepthook = sys.__excepthook__
    passed, total, lines = 0, 0, []

    def check(name, cond, note=""):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        lines.append(f"{'PASS' if cond else 'FAIL'} | {name}" + (f" ({note})" if note else ""))
        print(lines[-1], flush=True)

    rng = np.random.default_rng(18)
    n = 1200
    d, _ = r16.synth(n, rng, "e", effect=-0.3)
    d["sub"]["cc"] = rng.random(n) < 0.8
    d["hosp_death"] = (rng.random(n) < 0.12).astype(int)
    for rep in (0, 7):
        idx = np.random.default_rng(SEED + rep).integers(0, n, n)
        bp, rp, ref = draw(d, "pat", rep), r16.draw(d, "e", "pat", rep), asd.take(d, idx)
        check(f"patient-level draw {rep} equals rng.integers(0, n, n) and r16.draw(pat); hospital numbers kept",
              np.array_equal(bp["X"], ref["X"]) and np.array_equal(bp["A"], ref["A"]) and np.array_equal(bp["X"], rp["X"])
              and np.array_equal(bp["site"], d["site"][idx]))
        bh, rh = draw(d, "hosp", rep), asd.resample(d, "e", np.random.default_rng(SEED + rep))
        check(f"hospital-level draw {rep} equals asd.resample", np.array_equal(bh["X"], rh["X"]) and np.array_equal(bh["A"], rh["A"])
              and np.array_equal(bh["site"], rh["site"]))
    i7, c7 = r7.draw(d, "m", np.random.default_rng(SEED + 3))
    i10, c10 = r10.draw(d, "m", np.random.default_rng(SEED + 3))
    ref = np.random.default_rng(SEED + 3).integers(0, n, n)
    check("r7.draw and r10.draw by patient equal rng.integers(0, n, n), without hospital copies",
          np.array_equal(i7, ref) and np.array_equal(i10, ref) and c7 is None and c10 is None)
    states = synth_states(d, rng)
    for p in ORDER:
        st = states[p]
        res, cache = PIPES[p]["point"](st)
        for scheme in ("hosp", "pat"):
            v1 = np.asarray(PIPES[p]["vec"](st, scheme, 1, cache), float)
            v2 = np.asarray(PIPES[p]["vec"](st, scheme, 1, cache), float)
            check(f"{p} {scheme}: {len(PIPES[p]['cols'])} columns, finite, reproducible",
                  v1.size == len(PIPES[p]["cols"]) and np.isfinite(v1).sum() >= max(1, v1.size - 2) and np.allclose(v1, v2, equal_nan=True),
                  f"{int(np.isfinite(v1).sum())} finite")
        rj = asd.to_json(res)
        bad = []
        for x in [i for i in ITEMS + CHECKS if i["pipe"] == p]:
            try:
                v = x["point"](rj)
                nv = x["n"](rj) if x["n"] else (1, 1)
                ok = v is None or isinstance(v, float)
                ok &= all(isinstance(k, int) for k in nv)
                c = PIPES[p]["cols"]
                sp = x["spec"]
                ok &= (sp[1] in c) if sp[0] == "lor" else ((sp[1] in c and sp[2] in c) if sp[0] == "ratio" else (sp[1] + ":A" in c))
            except Exception as e:
                ok = False
                bad.append(f"{x['id']} {type(e).__name__}")
            if not ok and not bad:
                bad.append(x["id"])
        check(f"{p}: point-estimate getters and interval columns of every item", not bad, "; ".join(bad))
    rr = np.random.default_rng(5)
    a, b = rr.normal(0.1, 0.2, 400), rr.normal(-0.2, 0.1, 400)
    rA, rB = rr.uniform(0.1, 0.3, 400), rr.uniform(0.1, 0.3, 400)
    a[:7] = np.nan
    B = {"cols": ["a", "b", "x:A", "x:B"], "v": np.column_stack([a, b, rA, rB])}
    q = interval(("lor", "a"), B)
    check("interval of a log column: exp of the 2.5th and 97.5th percentiles of the finite values",
          q is not None and abs(q[0] - math.exp(np.percentile(a[7:], 2.5))) < 1e-12 and q[2] == 393)
    q = interval(("ratio", "a", "b"), B)
    dlt = (a - b)[np.isfinite(a - b)]
    check("interval of a ratio: within-replicate log difference", q is not None and abs(q[1] - math.exp(np.percentile(dlt, 97.5))) < 1e-12)
    lorx = np.log(rA / (1 - rA)) - np.log(rB / (1 - rB))
    q = interval(("risk", "x", "OR"), B)
    q2 = interval(("risk", "x", "RD"), B)
    check("interval from replicate risks (OR and RD), as r16.risk_entry", abs(q[0] - math.exp(np.percentile(lorx, 2.5))) < 1e-12
          and abs(q2[1] - np.percentile(rA - rB, 97.5)) < 1e-12)
    B2 = {"cols": ["a"], "v": np.r_[np.full(960, np.nan), np.ones(40)][:, None]}
    check("fewer than 50 valid replicates: no interval", interval(("lor", "a"), B2) is None)
    cases = [("or", 0.60, 0.95, "favours"), ("or", 0.80, 1.20, "benefit_excluded"), ("or", 0.75, 1.20, "inconclusive"),
             ("or", 1.05, 1.40, "opposite"), ("or", 0.70, 1.00, "inconclusive"), ("or", 0.76, 1.00, "benefit_excluded"),
             ("or_dose", 0.50, 0.90, "favours_dose"), ("nc", 0.80, 1.30, "no_signal"), ("nc", 1.01, 1.30, "signal"), ("nc", 0.5, 0.99, "signal"),
             ("ap", 0.64, 0.98, "fewer"), ("ap", 0.80, 1.20, "no_direction"), ("ap", 1.10, 1.50, "ap_opposite"), ("hr", 0.6, 0.9, "association"),
             ("hr", 0.8, 1.1, "unclear"), ("ratio", 1.04, 2.1, "higher"), ("ratio", 0.65, 0.98, "lower"), ("ratio", 0.9, 1.2, "no_difference"),
             ("rr", 0.6, 0.9, "not_read"), ("or", None, None, "no_interval")]
    for kind, lo, hi, want in cases:
        check(f"reading {kind} ({lo}, {hi}) -> {want}", read(kind, lo, hi) == want)
    check("threshold flags: 0.7534 and 0.9951 flagged for an OR, 0.7534 not for a negative control",
          len(at_threshold("or", 0.7534, 0.9951)) == 2 and len(at_threshold("nc", 0.7534, 1.2)) == 0)
    check("counts masked as the project does", sc.mask_n(7) == "<10" and sc.mask_n(12) == "10" and sc.mask_n(4605) == "4605" and sc.mask_n(0) == "0")
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "x.txt")
        open(f, "wb").write(b"r18")
        check("SHA-256 of a file", sha(f) == hashlib.sha256(b"r18").hexdigest())
    check("every batch-1 item has a reading kind and a unique id", len({x["id"] for x in ITEMS}) == 39 and all(x["kind"] in THRESH for x in ITEMS))
    lines.append(f"selftest: {passed}/{total} passed")
    print(lines[-1], flush=True)
    if args.out:
        open(args.out, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    if passed != total:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("selftest")
    x.add_argument("--out")
    x = s.add_parser("extract")
    x.add_argument("--private", required=True)
    x.add_argument("--mimic", required=True)
    x.add_argument("--eicu", required=True)
    for name in ("estimate", "gate", "report"):
        x = s.add_parser(name)
        x.add_argument("--private", required=True)
        x.add_argument("--results", required=True)
        if name == "report":
            x.add_argument("--overwrite", action="store_true")
    x = s.add_parser("boot")
    x.add_argument("--pipe", choices=list(PIPES), required=True)
    x.add_argument("--scheme", choices=["hosp", "pat"], required=True)
    x.add_argument("--start", type=int, required=True)
    x.add_argument("--stop", type=int, required=True)
    x.add_argument("--private", required=True)
    x = s.add_parser("run")
    x.add_argument("--scheme", choices=["hosp", "pat"], required=True)
    x.add_argument("--private", required=True)
    x.add_argument("--workers", type=int, default=22)
    a = ap.parse_args()
    {"selftest": selftest, "extract": extract, "estimate": estimate, "boot": boot, "run": run, "gate": gate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
