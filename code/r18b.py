#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

sys.dont_write_bytecode = True
import argparse
import glob
import json
import math
import re
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import r18
from r18 import asd, e1, r7, r11, r14, r16, sc
import e2_burden as e2

MS = os.path.join(ROOT, "manuscript_draft")
AF1 = "Supplementary_results_v13.md"
MAIN = "Manuscript_full_draft_v18.md"
MIN_BOOT = asd.MIN_BOOT
NEAR = 0.010


lor, draw, pr, say = r18.lor, r18.draw, r18.pr, r18.say


def record(P, step, t0, start, extra=None):
    f = pr(P, "timing_b2.json")
    T = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {"steps": []}
    T["steps"].append(dict({"step": step, "start": start, "end": r18.now_iso(), "seconds": round(time.time() - t0, 1)}, **(extra or {})))
    json.dump(T, open(f, "w", encoding="utf-8"), indent=1)


def asd2_vec(st, scheme, rep, cache):
    res = asd.run_stage2(draw(st["d"], scheme, rep), 2, "e", dict(cache))
    return [lor(res.get("primary")), res["iv"]["rd"]]


ASD1_COLS = ["primary", "negctl", "del72", "hyp", "pp", "iptw", "cc", "rass3", "egfr50", "crude", "death", "iv_rd"]


def asd1_load(P, est=None):
    df, flow = asd.load_key(P, "c1e", stage2=True)
    return {"d": asd.build(df, "e", 1, stage2=True), "flow": flow}


def asd1_point(st):
    cache = {}
    return asd.run_stage2(st["d"], 1, "e", cache), cache


def asd1_vec(st, scheme, rep, cache):
    res = asd.run_stage2(draw(st["d"], scheme, rep), 1, "e", dict(cache))
    return [lor(res.get(n)) for n in ASD1_COLS[:-1]] + [res["iv"]["rd"]]


def r7b_vec(st, scheme, rep, cache):
    res = r7.replicate(st["dd"], "c2e", "e" if scheme == "hosp" else "m", rep, cache)
    return [lor(res.get(n)) for n in ("m0_primary", "px_icu_oq")]


R7C1_COLS = ["m0_primary", "m0_negctl", "m1_primary", "m2_primary", "m1_negctl", "m2_negctl"]


def r7k_load(key):
    def load(P, est=None):
        df, flow, dd = r7.load_dd(P, key)
        st = {"dd": dd, "d": dd["m0"], "flow": flow}
        if est is None:
            st["df"] = df
        return st
    return load


def r7k_point(key):
    def point(st):
        cache = {}
        res = r7.analyses(st["dd"], key, cache)
        for m, d in st["dd"].items():
            r7.balance(d, cache, m)
        if key in r7.BASE_KEYS and "df" in st:
            r7.describe_new(st["df"], st["dd"], cache)
        return res, cache
    return point


def r7k_vec(key, cols):
    def vec(st, scheme, rep, cache):
        res = r7.replicate(st["dd"], key, "e" if scheme == "hosp" else "m", rep, cache)
        return [lor(res.get(n)) for n in cols]
    return vec


R11B_COLS = ["primary", "screen_scr_any", "screen_scr_ok", "screen_blk_ok_share"]


def r11b_vec(st, scheme, rep, cache):
    v = r11.flat(r11.analyses_all(draw(st["d"], scheme, rep), "e", dict(cache)), "e")
    names = r11.flat_names("e")
    return [v[names.index(n)] for n in R11B_COLS]


C1B_NAMES = ["primary", "negctl", "kadj_negctl", "kexc_negctl"]


def c1b_vec(st, scheme, rep, cache):
    res = r16.est_key(r16.draw(st["d"], "e", "main" if scheme == "hosp" else "pat", rep), "c1e", dict(cache))
    return [res[n][k] for n in C1B_NAMES for k in ("riskA", "riskB")]


E1B_COLS = ["primary", "base:yes", "base:no:ratio", "surg:no", "surg:yes", "surg:yes:ratio", "hosp:high", "hosp:mid", "hosp:low",
            "hosp:mid:ratio", "hosp:low:ratio", "rass:moderate:ratio", "rass:missing:ratio", "discharge", "per_assessment"]


def e1b_vec(st, scheme, rep, cache):
    res, _ = e1.all_estimates(draw(st["d"], scheme, rep), "e", st["shares_m"])
    names, v = e1.flat(res, "e")
    return [v[names.index(n)] for n in E1B_COLS]


R14B_COLS = ["main_primary", "main_ic", "main_negctl", "pre_ic", "pre_negctl"]


def r14b_vec(st, scheme, rep, cache):
    v = r14.flat(r14.estimates(draw(st["d"], scheme, rep), dict(cache)))
    return [v[r14.NAMES.index(n)] for n in R14B_COLS]


def e2_load(P, est=None):
    d, flow = e2.load_arrays(P, "c1e")
    return {"d": d, "flow": flow}


def e2_point(st):
    cache = {}
    return e2.estimates(st["d"], cache), cache


def e2_vec(st, scheme, rep, cache):
    e = e2.estimates(draw(st["d"], scheme, rep), dict(cache))
    return [e[o]["diff"] for o in e2.OUTS]


PIPES = {
    "asd2": dict(script="asd.py", nboot=2000, chunk=125, cols=["primary", "iv_rd"], load=r18.asd_load, point=r18.asd_point, vec=asd2_vec),
    "r7b": dict(script="r7.py", nboot=2000, chunk=125, cols=["m0_primary", "px_icu_oq"], load=r18.r7_load, point=r18.r7_point, vec=r7b_vec),
    "r11b": dict(script="r11.py", nboot=2000, chunk=125, cols=R11B_COLS, load=r18.r11_load, point=r18.r11_point, vec=r11b_vec),
    "c1b": dict(script="r16.py", nboot=2000, chunk=125, cols=[f"{n}:{a}" for n in C1B_NAMES for a in ("A", "B")], load=r18.c1_load,
                point=r18.c1_point, vec=c1b_vec),
    "e1b": dict(script="explore_ppi.py", nboot=2000, chunk=125, cols=E1B_COLS, load=r18.e1_load, point=r18.e1_point, vec=e1b_vec),
    "r14b": dict(script="r14.py", nboot=2000, chunk=125, cols=R14B_COLS, load=r18.r14_load, point=r18.r14_point, vec=r14b_vec),
    "asd1": dict(script="asd.py", nboot=2000, chunk=250, cols=ASD1_COLS, load=asd1_load, point=asd1_point, vec=asd1_vec),
    "r7c1": dict(script="r7.py", nboot=2000, chunk=250, cols=R7C1_COLS, load=r7k_load("c1e"), point=r7k_point("c1e"), vec=r7k_vec("c1e", R7C1_COLS)),
    "r7dose": dict(script="r7.py", nboot=2000, chunk=250, cols=["m0_primary", "m0_negctl"], load=r7k_load("dose"), point=r7k_point("dose"),
                   vec=r7k_vec("dose", ["m0_primary", "m0_negctl"])),
    "e2c1": dict(script="e2_burden.py", nboot=2000, chunk=250, cols=list(e2.OUTS), load=e2_load, point=e2_point, vec=e2_vec),
}
ORDER = ["r14b", "r11b", "asd2", "e1b", "r7b", "c1b", "asd1", "r7c1", "r7dose", "e2c1"]
STRATA16 = ("moderate", "deep", "missing")
UNITS16 = ("rass3", "light", "moderate", "deep", "missing")


def load_stored(R):
    J = lambda f: json.load(open(os.path.join(R, f), encoding="utf-8"))
    c2, c1, r16j = J("ppi_estimates.json")["per_db"]["e"], J("famdose_estimates.json")["per_db"]["e"], J("r16.json")
    return {"c2": c2["analyses"], "c2iv": c2["iv"], "c1": c1["analyses"], "c1iv": c1["iv"], "r7": J("r7.json")["per_key"],
            "r11": J("r11.json")["db"]["e"], "c1r16": r16j["famdose"]["e"]["est"], "inter": r16j["interaction"]["e"]["vs_light"],
            "scr": r16j["db"]["e"]["scr"], "pat16": r16j["db"]["e"]["est_pat"], "icpat16": r16j["db"]["e"]["ic_pat"],
            "est16": r16j["db"]["e"]["est"], "e1": J("ppi_explore_E1.json")["db"]["e"], "r14": J("r14.json"),
            "e2": J("e2_burden.json")["famdose"]["e"], "r18": J("r18.json")}


def _or(e, k="OR"):
    return (e[k], e[k + "_lo"], e[k + "_hi"])


def _lohi(e, k):
    return (e[k], e["lo"], e["hi"])


def _ci(e, k="OR", c="CI"):
    return (e[k], e[c][0], e[c][1])


def nn(e, a="nA", b="nB"):
    return (e[a], e[b])


def item(i, pipe, kind, label, key, stored, point, spec, n=None, dp=4, tol=None):
    return dict(id=i, pipe=pipe, kind=kind, label=label, key=key, stored=stored, point=point, spec=spec, n=n, dp=dp, tol=tol,
                script=PIPES[pipe]["script"] if pipe in PIPES else "r16.py")


def e1_or(i, s, v, label):
    return item(i, "e1b", "or", label, f"ppi_explore_E1.json db.e.strata.{s}.{v}", lambda S, s=s, v=v: _ci(S["e1"]["strata"][s][v]),
                lambda r, s=s, v=v: r["strata"][s][v]["OR"], ("lor", f"{s}:{v}"), lambda r, s=s, v=v: nn(r["strata"][s][v]))


def e1_ratio(i, s, v, ref, label):
    return item(i, "e1b", "ratio", label, f"ppi_explore_E1.json db.e.strata.{s}.{v}.ratio_vs_ref",
                lambda S, s=s, v=v: (S["e1"]["strata"][s][v]["ratio_vs_ref"],) + tuple(S["e1"]["strata"][s][v]["ratio_CI"]),
                lambda r, s=s, v=v, f=ref: r["strata"][s][v]["OR"] / r["strata"][s][f]["OR"], ("lor", f"{s}:{v}:ratio"))


def asd1_it(i, name, label, kind="or_dose", dp=4):
    return item(i, "asd1", kind, label, f"famdose_estimates.json per_db.e.analyses.{name}", lambda S, m=name: _or(S["c1"][m]),
                lambda r, m=name: r[m]["OR"], ("lor", name), lambda r, m=name: nn(r[m]), dp=dp)


def r7k_it(i, pipe, key, name, label, kind):
    return item(i, pipe, kind, label, f"r7.json per_key.{key}.{name}", lambda S, k=key, m=name: _or(S["r7"][k][m]),
                lambda r, m=name: r[m]["OR"], ("lor", name), lambda r, m=name: nn(r[m]))


def scr11(i, k, label):
    return item(i, "r11b", "ratio", label, f"r11.json db.e.screen.{k}", lambda S, k=k: _lohi(S["r11"]["screen"][k], "ratio"),
                lambda r, k=k: r["screen"][k]["ratio"], ("lor", f"screen_{k}"))


def c1b_it(i, name, label):
    return item(i, "c1b", "nc", label, f"r16.json famdose.e.est.{name}", lambda S, m=name: _or(S["c1r16"][m]), lambda r, m=name: r[m]["OR"],
                ("risk", name, "OR"), lambda r, m=name: nn(r[m]))


def e2_it(i, o, label, kind):
    return item(i, "e2c1", kind, label, f"e2_burden.json famdose.e.{o}", lambda S, o=o: tuple(round(S["e2"][o][k], 4) for k in ("diff", "lo", "hi")),
                lambda r, o=o: r[o]["diff"], ("raw", o),
                (lambda r: (r["n"]["A_rass3"], r["n"]["B_rass3"])) if o.endswith("rass3") else (lambda r: (r["n"]["A"], r["n"]["B"])), dp=4)


def pair16(i, u, scale, label):
    kind = "diff" if scale == "RD" else "ratio"
    return item(i, "r16s", kind, label, f"r16.json interaction.e.vs_light.{u}.{scale}",
                lambda S, u=u, s=scale: (S["inter"][u][s]["est"], S["inter"][u][s]["lo"], S["inter"][u][s]["hi"]),
                lambda r, u=u, s=scale: r["pair"][u][s], ("pair", f"{u}_primary", "light_primary", scale))


def scr16(i, u, label):
    return item(i, "r16s", "ratio", label, f"r16.json db.e.scr.scr_{u}", lambda S, u=u: _lohi(S["scr"][f"scr_{u}"], "ratio"),
                lambda r, u=u: r["scr"][u], ("lor", f"scr_{u}"))


ITEMS2 = [
    item("B2-01", "asd2", "iv", "PPI comparison instrumental-variable risk difference (hospital preference)", "ppi_estimates.json per_db.e.iv", lambda S: (S["c2iv"]["rd"], S["c2iv"]["rd_lo"], S["c2iv"]["rd_hi"]),
         lambda r: r["iv"]["rd"], ("raw", "iv_rd")),
    r7k_it("B2-02", "r7b", "c2e", "px_icu_oq", "First quetiapine in the ICU", "ap"),
    scr11("B2-03", "scr_any", "Ratio of screens per ICU day before the first positive"), scr11("B2-04", "scr_ok", "Ratio of assessable screens per ICU day before the first positive"),
    scr11("B2-05", "blk_ok_share", "Ratio of 12-h periods with an assessable screen"),
    c1b_it("B2-06", "negctl", "Dose comparison, negative control"), c1b_it("B2-07", "kadj_negctl", "Dose comparison, negative control, kidney covariates added"),
    c1b_it("B2-08", "kexc_negctl", "Dose comparison, negative control, patients on kidney replacement therapy removed"),
    e1_or("B2-09", "base", "yes", "Delirium assessment in the 24 h before time zero"), e1_ratio("B2-10", "base", "no", "yes", "No assessment ÷ assessment"),
    e1_or("B2-11", "surg", "no", "ICU admission not from the operating room"), e1_or("B2-12", "surg", "yes", "ICU admission from the operating room"), e1_ratio("B2-13", "surg", "yes", "no", "Operating room ÷ other"),
    e1_or("B2-14", "hosp", "high", "Hospital screening intensity high"), e1_or("B2-15", "hosp", "mid", "Hospital screening intensity middle"), e1_or("B2-16", "hosp", "low", "Hospital screening intensity low"),
    e1_ratio("B2-17", "hosp", "mid", "high", "Middle ÷ high"), e1_ratio("B2-18", "hosp", "low", "high", "Low ÷ high"),
    e1_ratio("B2-19", "rass", "moderate", "light", "Moderate sedation ÷ awake or light (run of E1)"),
    e1_ratio("B2-20", "rass", "missing", "light", "RASS not recorded ÷ awake or light (run of E1)"),
    item("B2-21", "e1b", "ratio", "Odds ratio of ICU discharge alive within 168 h (PPI vs famotidine)", "ppi_explore_E1.json db.e.discharge", lambda S: _ci(S["e1"]["discharge"]),
         lambda r: r["discharge"]["OR"], ("lor", "discharge")),
    item("B2-22", "e1b", "ratio", "Ratio of detection per assessment (PPI vs famotidine)", "ppi_explore_E1.json db.e.per_assessment", lambda S: _ci(S["e1"]["per_assessment"], "RR"),
         lambda r: r["per_assessment"]["RR"], ("lor", "per_assessment")),
    item("B2-23", "r14b", "hr", "Interval-censored hazard ratio of the pre-time-zero model", "r14.json est.pre_ic", lambda S: _lohi(S["r14"]["est"]["pre_ic"], "HR"),
         lambda r: r["pre_ic"]["HR"], ("lor", "pre_ic"), lambda r: nn(r["pre_ic"], "n_inf_A", "n_inf_B"), tol=0.002),
    item("B2-24", "r14b", "ratio", "Interval-censored hazard ratio: pre-time-zero model ÷ main model", "r14.json ratio.ic_pre_vs_main", lambda S: _lohi(S["r14"]["ratio"]["ic_pre_vs_main"], "ratio"),
         lambda r: r["pre_ic"]["HR"] / r["main_ic"]["HR"], ("ratio", "pre_ic", "main_ic"), tol=0.002),
    item("B2-25", "r14b", "nc", "Negative control, pre-time-zero model", "r14.json est.pre_negctl", lambda S: _lohi(S["r14"]["est"]["pre_negctl"], "OR"),
         lambda r: r["pre_negctl"]["OR"], ("lor", "pre_negctl"), lambda r: nn(r["pre_negctl"])),
    item("B2-26", "r14b", "ratio", "Negative control: pre-time-zero model ÷ main model", "r14.json ratio.negctl_pre_vs_main", lambda S: _lohi(S["r14"]["ratio"]["negctl_pre_vs_main"], "ratio"),
         lambda r: r["pre_negctl"]["OR"] / r["main_negctl"]["OR"], ("ratio", "pre_negctl", "main_negctl")),
    asd1_it("B2-27", "del72", "Dose comparison, 72 h"), asd1_it("B2-28", "hyp", "Dose comparison, censored at ICU discharge"), asd1_it("B2-29", "pp", "Dose comparison, per protocol"),
    asd1_it("B2-30", "iptw", "Dose comparison, inverse probability of treatment weighting"), asd1_it("B2-31", "cc", "Dose comparison, complete cases"), asd1_it("B2-32", "rass3", "Dose comparison, RASS ≥ −3"),
    asd1_it("B2-33", "egfr50", "Dose comparison, eGFR <50"), asd1_it("B2-34", "crude", "Dose comparison, unweighted (descriptive)", dp=2), asd1_it("B2-35", "death", "Dose comparison, in-hospital death"),
    item("B2-36", "asd1", "iv", "Dose comparison instrumental-variable risk difference", "famdose_estimates.json per_db.e.iv", lambda S: (S["c1iv"]["rd"], S["c1iv"]["rd_lo"], S["c1iv"]["rd_hi"]),
         lambda r: r["iv"]["rd"], ("raw", "iv_rd")),
    r7k_it("B2-37", "r7c1", "c1e", "m1_primary", "Dose comparison, additional adjustment M1", "or_dose"), r7k_it("B2-38", "r7c1", "c1e", "m2_primary", "Dose comparison, additional adjustment M2", "or_dose"),
    r7k_it("B2-39", "r7c1", "c1e", "m1_negctl", "Dose comparison, negative control, M1", "nc"), r7k_it("B2-40", "r7c1", "c1e", "m2_negctl", "Dose comparison, negative control, M2", "nc"),
    r7k_it("B2-41", "r7dose", "dose", "m0_primary", "Dose comparison, other classification of eICU order doses", "or_dose"),
    r7k_it("B2-42", "r7dose", "dose", "m0_negctl", "Dose comparison, negative control, other classification", "nc"),
    e2_it("B2-43", "dcfd7", "Difference in days alive and free of delirium or coma", "burden_pos"), e2_it("B2-44", "deldays7", "Difference in days with delirium in the ICU", "burden_neg"),
    e2_it("B2-45", "dcfd7_rass3", "Difference in days alive and free of delirium or coma, RASS ≥ −3", "burden_pos"),
] + [pair16(f"B2-{46 + 3 * j + k}", u, s, f"{lab} ÷ awake or light (run of R16), {sl}")
     for j, (s, sl) in enumerate((("OR", "ratio of odds ratios"), ("RR", "ratio of risk ratios"), ("RD", "difference in risk differences")))
     for k, (u, lab) in enumerate((("moderate", "Moderately sedated"), ("deep", "Deeply sedated"), ("missing", "RASS not recorded")))] + [
    scr16(f"B2-{55 + k}", u, f"Ratio of assessable screens per ICU day, {lab}") for k, (u, lab) in
    enumerate((("rass3", "RASS ≥ −3"), ("light", "awake or light"), ("moderate", "moderate"), ("deep", "deep"), ("missing", "RASS not recorded")))]

CHECKS2 = [
    item("c-asd2-p", "asd2", "or", "primary", "", lambda S: _or(S["c2"]["primary"]), lambda r: r["primary"]["OR"], ("lor", "primary")),
    item("c-r7b-p", "r7b", "or", "m0_primary", "", lambda S: _or(S["r7"]["c2e"]["m0_primary"]), lambda r: r["m0_primary"]["OR"], ("lor", "m0_primary")),
    item("c-r11b-p", "r11b", "or", "primary", "", lambda S: _lohi(S["r11"]["primary"], "OR"), lambda r: r["primary"]["OR"], ("lor", "primary")),
    item("c-c1b-p", "c1b", "or", "primary", "", lambda S: _or(S["c1r16"]["primary"]), lambda r: r["primary"]["OR"], ("risk", "primary", "OR")),
    item("c-e1b-p", "e1b", "or", "primary", "", lambda S: _ci(S["e1"]["primary"]), lambda r: r["primary"]["OR"], ("lor", "primary")),
    item("c-r14b-p", "r14b", "or", "main_primary", "", lambda S: _lohi(S["r14"]["est"]["main_primary"], "OR"), lambda r: r["main_primary"]["OR"], ("lor", "main_primary")),
    item("c-r14b-ic", "r14b", "hr", "main_ic", "", lambda S: _lohi(S["r14"]["est"]["main_ic"], "HR"), lambda r: r["main_ic"]["HR"], ("lor", "main_ic"), tol=0.002),
    item("c-r14b-n", "r14b", "nc", "main_negctl", "", lambda S: _lohi(S["r14"]["est"]["main_negctl"], "OR"), lambda r: r["main_negctl"]["OR"], ("lor", "main_negctl")),
    item("c-asd1-p", "asd1", "or", "primary", "", lambda S: _or(S["c1"]["primary"]), lambda r: r["primary"]["OR"], ("lor", "primary")),
    item("c-asd1-n", "asd1", "or", "negctl", "", lambda S: _or(S["c1"]["negctl"]), lambda r: r["negctl"]["OR"], ("lor", "negctl")),
    item("c-r7c1-p", "r7c1", "or", "m0_primary", "", lambda S: _or(S["r7"]["c1e"]["m0_primary"]), lambda r: r["m0_primary"]["OR"], ("lor", "m0_primary")),
    item("c-r7c1-n", "r7c1", "or", "m0_negctl", "", lambda S: _or(S["r7"]["c1e"]["m0_negctl"]), lambda r: r["m0_negctl"]["OR"], ("lor", "m0_negctl")),
]


def batch1_pat(S, i):
    row = next(r for r in S["r18"]["batch1"] if r["id"] == i)
    return row["patient"]["lo"], row["patient"]["hi"]


PAT_REF2 = [("asd2", ("lor", "primary"), lambda S: (S["pat16"]["primary"]["OR_lo"], S["pat16"]["primary"]["OR_hi"]), None, "R16 est_pat primary"),
            ("r7b", ("lor", "m0_primary"), lambda S: (S["pat16"]["primary"]["OR_lo"], S["pat16"]["primary"]["OR_hi"]), None, "R16 est_pat primary"),
            ("r11b", ("lor", "primary"), lambda S: (S["pat16"]["primary"]["OR_lo"], S["pat16"]["primary"]["OR_hi"]), None, "R16 est_pat primary"),
            ("e1b", ("lor", "primary"), lambda S: (S["pat16"]["primary"]["OR_lo"], S["pat16"]["primary"]["OR_hi"]), 0.002, "R16 est_pat primary"),
            ("r14b", ("lor", "main_primary"), lambda S: (S["pat16"]["primary"]["OR_lo"], S["pat16"]["primary"]["OR_hi"]), None, "R16 est_pat primary"),
            ("r14b", ("lor", "main_negctl"), lambda S: (S["pat16"]["negctl"]["OR_lo"], S["pat16"]["negctl"]["OR_hi"]), None, "R16 est_pat negctl"),
            ("r14b", ("lor", "main_ic"), lambda S: (S["icpat16"]["ic_all"]["lo"], S["icpat16"]["ic_all"]["hi"]), 0.002, "R16 ic_pat ic_all"),
            ("c1b", ("risk", "primary", "OR"), lambda S: batch1_pat(S, "B1-23"), None, "R18 batch 1, B1-23"),
            ("asd1", ("lor", "primary"), lambda S: batch1_pat(S, "B1-23"), None, "R18 batch 1, B1-23"),
            ("r7c1", ("lor", "m0_primary"), lambda S: batch1_pat(S, "B1-23"), None, "R18 batch 1, B1-23")]


def interval(spec, B):
    if B is None:
        return None
    if spec[0] == "raw":
        v = r18.bcol(B, spec[1])
        v = v[np.isfinite(v)]
        if v.size < MIN_BOOT:
            return None
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)
    if spec[0] == "pair":
        a = r16.scales(r18.bcol(B, spec[1] + ":A"), r18.bcol(B, spec[1] + ":B"))[spec[3]]
        b = r16.scales(r18.bcol(B, spec[2] + ":A"), r18.bcol(B, spec[2] + ":B"))[spec[3]]
        v = (a - b)[np.isfinite(a - b)]
        if v.size < MIN_BOOT:
            return None
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        return ((math.exp(lo), math.exp(hi)) if spec[3] != "RD" else (lo, hi)) + (int(v.size),)
    return r18.interval(spec, B)


RULE = {"or": "Methods: upper <1 favours PPI; lower >1 opposite; includes 1 with lower >0.75 benefit excluded; otherwise inconclusive; "
              "lower within 0.01 of 0.75 (interval including 1) described by range",
        "nc": "negative control: excludes 1 signal, includes 1 no signal",
        "ap": "antipsychotic outcome, one database: direction only (Additional file 1)",
        "hr": "interval-censored HR: upper <1 association, lower >1 opposite, otherwise unclear (R16 section 5)",
        "ratio": "ratio: excludes 1 (above or below) or includes 1",
        "diff": "difference of risk differences: excludes 0 or includes 0",
        "burden": "burden outcome: 1-day rule of addendum E2",
        "none": "not read"}
RULE["or_dose"] = RULE["or"].replace("favours PPI", "favours <=20 mg/day")
LAB = {"favours": "favours PPI", "favours_dose": "favours <=20 mg/day", "benefit_excluded": "clinically important benefit excluded",
       "opposite": "opposite direction", "inconclusive": "inconclusive", "near_0.75": "lower limit close to 0.75: described by its range",
       "signal": "signal of residual confounding", "no_signal": "no signal", "fewer": "fewer with PPI", "ap_opposite": "opposite direction",
       "no_direction": "no direction shown", "association": "association (favours PPI)", "hr_opposite": "opposite direction", "unclear": "unclear",
       "excl_above": "excludes 1 (above)", "excl_below": "excludes 1 (below)", "incl_1": "includes 1", "excl0_above": "excludes 0 (above)",
       "excl0_below": "excludes 0 (below)", "incl_0": "includes 0", "not_read": "not read", "no_interval": "no interval"}


def read_cell(kind, lo, hi):
    if kind in ("iv", "rr", "rd"):
        return "not_read"
    if lo is None or hi is None:
        return "no_interval"
    if kind in ("or", "or_dose"):
        if hi < 1:
            return "favours_dose" if kind == "or_dose" else "favours"
        if lo > 1:
            return "opposite"
        if abs(lo - asd.MCID) <= NEAR + 1e-12:
            return "near_0.75"
        return "benefit_excluded" if lo > asd.MCID else "inconclusive"
    if kind == "nc":
        return "signal" if (lo > 1 or hi < 1) else "no_signal"
    if kind == "ap":
        return "fewer" if hi < 1 else ("ap_opposite" if lo > 1 else "no_direction")
    if kind == "hr":
        return "association" if hi < 1 else ("hr_opposite" if lo > 1 else "unclear")
    if kind == "ratio":
        return "excl_above" if lo > 1 else ("excl_below" if hi < 1 else "incl_1")
    if kind == "diff":
        return "excl0_above" if lo > 0 else ("excl0_below" if hi < 0 else "incl_0")
    if kind in ("burden_pos", "burden_neg"):
        a, b = (lo, hi) if kind == "burden_pos" else (-hi, -lo)
        return "E2:" + e2.verdict({"lo": a, "hi": b, "I2": 0.0, "same_sign": True}, "dcfd7").split("; ", 1)[1]
    raise ValueError(kind)


def rule_of(kind):
    return RULE["burden"] if kind.startswith("burden") else RULE.get(kind, RULE["none"])


def lab_of(code):
    return code if code.startswith("E2:") else LAB[code]


def fnum(kind, v):
    if v is None:
        return "—"
    if kind == "rd":
        s = f"{100 * v:.1f}"
    elif kind == "diff":
        s = f"{100 * v:.1f}"
    elif kind == "iv":
        s = f"{v:.3f}"
    else:
        s = f"{v:.2f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def fspan(kind, lo, hi):
    if lo is None or hi is None:
        return "no interval"
    sep = " to " if kind in ("rd", "diff", "iv", "burden_pos", "burden_neg") else "–"
    return f"{fnum(kind, lo)}{sep}{fnum(kind, hi)}"


def ftext(kind, x, lo, hi):
    return f"{fnum(kind, x)} ({fspan(kind, lo, hi)})"


def T(sec, hdr, row, col, nth=0, lc=1, point=False):
    return {"type": "table", "file": AF1, "sec": sec, "hdr": hdr, "row": row, "col": col, "nth": nth, "lc": lc, "point": point}


def X(file, find, point=False):
    return {"type": "text", "file": file, "find": find, "point": point}


S3, S4H, S11D, S11E = "Table S3.", "eICU-CRD OR", "D. eICU-CRD", "E. eICU-CRD"
S3T = lambda lab, nth=0: T(S3, r"^eICU-CRD$", {0: "^" + re.escape(lab) + "$"}, "^eICU-CRD$", nth)
S4R = lambda lab, col, nth=0: T("Table S4.", S4H, {0: "^" + re.escape(lab) + "$"}, col, nth)
S9T = lambda lab: T("Table S9.", r"^eICU-CRD$", {0: "^" + re.escape(lab) + "$"}, "^eICU-CRD$")
S18T = lambda lab: T("Table S18.", r"^eICU-CRD: patients; OR \(95% CI\)$", {0: "^" + re.escape(lab) + "$"}, r"^eICU-CRD: patients; OR \(95% CI\)$")
S24T = lambda fac, stratum: T("Table S24.", r"^Holm-adjusted P$", {0: "^eICU-CRD$", 1: "^" + re.escape(fac) + "$", 2: "^" + re.escape(stratum) + "$"}, r"^OR \(95% CI\)$", lc=3)
S14T = lambda comp, col: T("Table S14.", r"^Ratio of ORs", {0: "^" + re.escape(comp) + "$", 1: "^eICU-CRD$"}, col, lc=2)
S6T = lambda outcome: T("Table S6.", r"^OR \(95% CI\)$", {0: r"^eICU-CRD \(subset", 1: "^" + re.escape(outcome) + "$"}, r"^OR \(95% CI\)$", lc=2)
S13D = lambda col: T("Table S13.", r"^RR \(95% CI\)$", {0: r"^Famotidine ≤20 vs ≥40", 1: "^eICU-CRD$"}, col, lc=2)
S23T = lambda excl: T("Table S23.", r"^First antipsychotic in the ICU, OR \(95% CI\)$", {0: "^eICU-CRD$", 1: "^" + re.escape(excl)}, r"^First antipsychotic in the ICU, OR \(95% CI\)$", lc=2)
S2T = lambda nth: T("Table S2.", r"^eICU-CRD PPI$", {0: r"^Ratio, PPI vs famotidine$"}, r"^eICU-CRD PPI$", nth)

WHERE = {
    "B1-01": [S3T("Follow-up censored at ICU discharge")],
    "B1-02": [S3T("Per-protocol")],
    "B1-03": [S3T("Inverse probability of treatment weighting")],
    "B1-04": [S3T("Complete cases")],
    "B1-05": [S3T("Unweighted"), T("Table S4.", r"^eICU-CRD without the block$", {0: r"^Unweighted / full model$"}, r"^eICU-CRD without the block$", point=True),
              X(AF1, "moved eICU-CRD from 0.74 to 0.97", True), X(MAIN, "Before adjustment the databases agreed (ORs 0.78 and 0.74)", True)],
    "B1-06": [S3T("In-hospital death")],
    "B1-07": [S3T("Additional adjustment: diabetes and insulin")],
    "B1-08": [S3T("Additional adjustment: diabetes, insulin and current-admission codes")],
    "B1-09": [S3T("Same, recorded RASS ≥ −3 at time zero")],
    "B1-10": [S3T("Additional adjustment: corticosteroids and sedative type")],
    "B1-11": [S3T("Gradient-boosted propensity score*")],
    "B1-12": [S3T("Doubly robust, gradient-boosted models*")],
    "B1-13": [S3T("Interval-censored hazard ratio, baseline in 6 intervals"), X(AF1, "0.91, 0.83–1.06")],
    "B1-14": [S3T("Additional adjustment: diabetes and insulin", 1)],
    "B1-15": [S3T("Additional adjustment: diabetes, insulin and current-admission codes", 1)],
    "B1-16": [S3T("Additional adjustment: corticosteroids and sedative type", 1)],
    "B1-17": [S3T("Gradient-boosted propensity score*", 1)],
    "B1-18": [S3T("Doubly robust, gradient-boosted models*", 1)],
    "B1-19": [S6T("Delirium (main definition)")],
    "B1-20": [S6T("First antipsychotic in the ICU (main)"), S23T("Applied"), X(AF1, "and 0.78 (0.64–0.98) with the exclusion"),
              X(MAIN, "eICU-CRD (0.78, 0.64–0.98)")],
    "B1-21": [X(AF1, "eICU-CRD 0.82 (0.65–0.98)")],
    "B1-22": [S23T("Not applied"), X(AF1, "0.77 (0.68–0.87) and 0.76 (0.63–0.94), against"), X(MAIN, "0.77 (0.68–0.87) and 0.76 (0.63–0.94) (Additional file 1: Table S23)")],
    "B1-23": [S9T("Main analysis"), S13D(r"^OR \(95% CI\)$"), S18T("Main analysis")],
    "B1-24": [S13D(r"^RR \(95% CI\)$")],
    "B1-25": [S13D(r"^RD, percentage points \(95% CI\)$")],
    "B1-26": [S18T("Chronic kidney disease and renal replacement therapy added")],
    "B1-27": [S18T("Without patients on renal replacement therapy")],
    "B1-28": [S18T("Bleeding-indication and PPI-infusion criteria applied")],
    "B1-29": [S4R("Invasive ventilation at time zero: no", 5), T("Table S11.", r"^Ratio, ventilated vs not$", {0: r"^APACHE first-day flag"}, r"^Not ventilated"),
              S24T("Invasive ventilation", "No")],
    "B1-30": [S4R("yes", 5, 1), T("Table S11.", r"^Ratio, ventilated vs not$", {0: r"^APACHE first-day flag"}, r"^Ventilated:"), S24T("Invasive ventilation", "Yes")],
    "B1-31": [S4R("yes", 6, 1), T("Table S11.", r"^Ratio, ventilated vs not$", {0: r"^APACHE first-day flag"}, r"^Ratio, ventilated vs not$")],
    "B1-32": [S4R("deep (≤ −4)", 6)],
    "B1-33": [X(AF1, "eICU-CRD 0.96 (0.88–1.22) in its own mix")],
    "B1-34": [X(AF1, "1.03 (0.89–1.34) in MIMIC-IV's"), X(AF1, "eICU-CRD in MIMIC-IV's mix 1.03, 0.89–1.34")],
    "B1-35": [T("Table S11.", r"^Restricted model$", {0: r"^Delirium, 7-day OR"}, r"^Restricted model$"), X(AF1, "(0.98, 0.90–1.14, against 0.97, 0.90–1.13)")],
    "B1-36": [T("Table S11.", r"^Restricted model$", {0: r"^Delirium, 7-day OR"}, r"^Ratio, restricted vs main$")],
    "B1-37": [T("Table S11.", r"^Ratio, ventilated vs not$", {0: r"^Time-stamped ventilator records"}, r"^Not ventilated")],
    "B1-38": [T("Table S11.", r"^Ratio, ventilated vs not$", {0: r"^Time-stamped ventilator records"}, r"^Ventilated:")],
    "B1-39": [T("Table S11.", r"^Ratio, ventilated vs not$", {0: r"^Time-stamped ventilator records"}, r"^Ratio, ventilated vs not$")],
    "B2-01": [X(AF1, "risk difference −0.071 (−0.168 to 0.212)")],
    "B2-02": [S6T("First quetiapine in the ICU"), X(AF1, "0.89 (0.65–1.19) in eICU-CRD (quetiapine only)")],
    "B2-03": [S2T(0), X(AF1, "1.14, 0.97–1.28")],
    "B2-04": [S2T(1), X(AF1, "all patients, 0.97 (0.93–1.01) and 1.13 (0.96–1.29)")],
    "B2-05": [S2T(2)],
    "B2-06": [S9T("Negative control: glucose <70 mg/dL"), S18T("Negative control, main analysis")],
    "B2-07": [S18T("Negative control, kidney covariates added")],
    "B2-08": [S18T("Negative control, without patients on renal replacement therapy")],
    "B2-09": [S4R("Any delirium assessment charted in the 24 h before time zero†: yes", 5), S24T("Delirium assessment in the 24 h before time zero", "Any"),
              X(AF1, "the estimate was 0.85 (0.71–0.98) in patients with any delirium assessment")],
    "B2-10": [S4R("no", 6), X(AF1, "(ratio 1.19, 1.00–1.94)")],
    "B2-11": [S4R("Surgical service or admitted from the operating room: no", 5), S24T("Admission from the operating room", "No")],
    "B2-12": [S4R("yes", 5, 0), S24T("Admission from the operating room", "Yes")],
    "B2-13": [S4R("yes", 6, 0)],
    "B2-14": [S4R("Hospital screening intensity (eICU-CRD): high (8 hospitals)", 5), S24T("Hospital's screening intensity", "Highest third"),
              X(AF1, "the eICU-CRD estimates were 1.07, 0.94 and 1.00 from high to low", True)],
    "B2-15": [S4R("middle (6 hospitals)", 5), S24T("Hospital's screening intensity", "Middle third")],
    "B2-16": [S4R("low (8 hospitals)", 5), S24T("Hospital's screening intensity", "Lowest third")],
    "B2-17": [S4R("middle (6 hospitals)", 6), X(AF1, "(ratios 0.88, 0.54–1.36")],
    "B2-18": [S4R("low (8 hospitals)", 6), X(AF1, "and 0.93, 0.58–1.36)")],
    "B2-19": [S4R("moderate (−3 to −2)", 6)],
    "B2-20": [S4R("not recorded", 6)],
    "B2-21": [X(AF1, "eICU-CRD 83% vs 85%, OR 0.87 (0.70–1.03)")],
    "B2-22": [X(AF1, "and 0.84 (0.74–1.01) in eICU-CRD")],
    "B2-23": [T("Table S11.", r"^Restricted model$", {0: r"^Hazard ratio, interval-censored model$"}, r"^Restricted model$")],
    "B2-24": [T("Table S11.", r"^Restricted model$", {0: r"^Hazard ratio, interval-censored model$"}, r"^Ratio, restricted vs main$")],
    "B2-25": [T("Table S11.", r"^Restricted model$", {0: r"^Negative control, 7-day OR$"}, r"^Restricted model$")],
    "B2-26": [T("Table S11.", r"^Restricted model$", {0: r"^Negative control, 7-day OR$"}, r"^Ratio, restricted vs main$")],
    "B2-27": [S9T("Delirium within 72 h")], "B2-28": [S9T("Follow-up censored at ICU discharge")], "B2-29": [S9T("Per-protocol")],
    "B2-30": [S9T("Inverse probability of treatment weighting")], "B2-31": [S9T("Complete cases")], "B2-32": [S9T("Recorded RASS ≥ −3 at time zero")],
    "B2-33": [S9T("eGFR <50 mL/min/1.73 m²")], "B2-34": [S9T("Unweighted")], "B2-35": [S9T("In-hospital death")],
    "B2-36": [X(AF1, "risk difference −0.003 (−0.354 to 0.399)")],
    "B2-37": [S9T("Additional adjustment: diabetes and insulin")], "B2-38": [S9T("Additional adjustment: diabetes, insulin and current-admission codes")],
    "B2-39": [S9T("Negative control, additional adjustment: diabetes and insulin")],
    "B2-40": [S9T("Negative control, additional adjustment: diabetes, insulin and current-admission codes")],
    "B2-41": [S9T("Alternative classification of eICU-CRD order doses")], "B2-42": [S9T("Negative control, alternative classification of eICU-CRD order doses")],
    "B2-43": [S9T("Days alive without delirium or coma, difference in days"), X(AF1, "eICU-CRD −0.05 (−0.29 to 0.20)")],
    "B2-44": [S9T("ICU days with delirium, difference in days"), X(AF1, "and 0.02 (−0.13 to 0.22)")],
    "B2-45": [S9T("Days alive without delirium or coma, recorded RASS ≥ −3, difference in days")],
    "B2-46": [S14T("Moderate (−3 to −2) vs awake or light", r"^Ratio of ORs")], "B2-47": [S14T("Deep (≤ −4) vs awake or light", r"^Ratio of ORs")],
    "B2-48": [S14T("RASS not recorded vs awake or light", r"^Ratio of ORs")],
    "B2-49": [S14T("Moderate (−3 to −2) vs awake or light", r"^Ratio of RRs")], "B2-50": [S14T("Deep (≤ −4) vs awake or light", r"^Ratio of RRs")],
    "B2-51": [S14T("RASS not recorded vs awake or light", r"^Ratio of RRs")],
    "B2-52": [S14T("Moderate (−3 to −2) vs awake or light", r"^Difference of RDs")], "B2-53": [S14T("Deep (≤ −4) vs awake or light", r"^Difference of RDs")],
    "B2-54": [S14T("RASS not recorded vs awake or light", r"^Difference of RDs")],
    "B2-55": [X(AF1, "recorded RASS ≥ −3, 0.97 (0.93–1.01) and 1.12 (1.00–1.23)")], "B2-56": [X(AF1, "awake or light, 0.96 (0.92–1.01) and 1.09 (0.97–1.17)")],
    "B2-57": [X(AF1, "moderate, 1.00 (0.88–1.13) and 1.40 (0.98–1.83)")], "B2-58": [X(AF1, "deep, 1.06 (0.93–1.22) and 1.01 (0.33–2.21)")],
    "B2-59": [X(AF1, "RASS not recorded, 0.83 (0.71–0.99) and 1.16 (0.91–1.57)")],
}
FIGURE = {"B1-20": "Figure 5 (figures/v14/Figure5_antipsychotic)", "B1-23": "Figure 3b (figures/v14/Figure3_delirium)",
          "B1-29": "Figure 4 (figures/r17/Figure4_effect_modification)", "B1-30": "Figure 4 (figures/r17/Figure4_effect_modification)",
          "B1-31": "Figure 4 (figures/r17/Figure4_effect_modification)", "B1-32": "Figure 4 (figures/r17/Figure4_effect_modification)"}


def md_lines(name):
    return open(os.path.join(MS, name), encoding="utf-8").read().splitlines()


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def tables_in(lines, sec):
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## ") and ln[3:].startswith(sec))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    out, i = [], start
    while i < end:
        if lines[i].startswith("|") and i + 1 < end and re.match(r"^\|[-|: ]+\|$", lines[i + 1].strip()):
            hdr = cells(lines[i])
            rows, j = [], i + 2
            prev = [""] * len(hdr)
            while j < end and lines[j].startswith("|"):
                c = cells(lines[j])
                ff = [c[k] if (c[k] or k > 2) else prev[k] for k in range(len(c))]
                rows.append((j + 1, c, ff))
                prev = ff
                j += 1
            out.append((hdr, rows))
            i = j
        else:
            i += 1
    return out


def sentence(line, pos):
    a = max(line.rfind(". ", 0, pos), line.rfind(".) ", 0, pos))
    a = 0 if a < 0 else a + 2
    b = line.find(". ", pos)
    b = len(line) if b < 0 else b + 1
    return line[a:b].strip()


def locate(spec, cache_lines):
    lines = cache_lines.setdefault(spec["file"], md_lines(spec["file"]))
    if spec["type"] == "text":
        hits = [(i, ln.find(spec["find"])) for i, ln in enumerate(lines) if spec["find"] in ln]
        if len(hits) != 1 or lines[hits[0][0]].count(spec["find"]) != 1:
            return {"error": f"substring found {sum(ln.count(spec['find']) for ln in lines)} times"}
        i, k = hits[0]
        head = next((lines[j].strip("# ").strip() for j in range(i, -1, -1) if lines[j].startswith("#")), "")
        return {"file": spec["file"], "where": head, "row": None, "column": None, "line": i + 1, "printed_text": spec["find"],
                "context": sentence(lines[i], k), "point_only": spec["point"]}
    for hdr, rows in tables_in(lines, spec["sec"]):
        if not any(re.search(spec["hdr"], h) for h in hdr):
            continue
        col = spec["col"] if isinstance(spec["col"], int) else next(k for k, h in enumerate(hdr) if re.search(spec["col"], h))
        found = [(ln, c, ff) for ln, c, ff in rows if all(re.search(rx, ff[k]) for k, rx in spec["row"].items())]
        if len(found) <= spec["nth"]:
            return {"error": "row not found"}
        ln, c, ff = found[spec["nth"]]
        head = next(lines[j][3:].split(". ")[0] for j in range(ln - 1, -1, -1) if lines[j].startswith("## "))
        return {"file": spec["file"], "where": head, "row": " / ".join(ff[:spec["lc"]]), "column": hdr[col], "line": ln, "printed_text": c[col],
                "context": "| " + " | ".join(c) + " |", "point_only": spec["point"]}
    return {"error": "table not found"}


def check_print(loc, kind, x, lo, hi):
    if "error" in loc:
        return False
    want = fnum(kind, x) if loc["point_only"] else (fspan(kind, lo, hi) if loc["printed_text"].count("(") == 0 or "," in loc["printed_text"]
                                                    else ftext(kind, x, lo, hi))
    return want in loc["printed_text"]


def extract(args):
    P = r18.priv(args)
    t0, start = time.time(), r18.now_iso()
    ns = argparse.Namespace
    for label, fn, files in (("R7 dose variant (eICU-CRD dose comparison, alternative dose classification)", lambda: r7.variant(ns(private=P, name="dose", results=None)),
                              ["e_dose_cohort", "e_dose_covars", "e_dose_follow", "e_dose_outcome"]),
                             ("E2 burden days, eICU-CRD", lambda: e2.extract(ns(private=P, db="e", results=None)), ["e_c1_dcfd", "e_c2_dcfd"])):
        if all(os.path.exists(os.path.join(P, f + ".parquet")) for f in files):
            say(f"{label}: already present, skipped")
            continue
        fn()
        say(f"{label}: written")
    record(P, "extract", t0, start)


def r16_point(P):
    E = json.load(open(os.path.join(P, "r16_est_c2e.json")))["res"]
    out = {"pair": {}, "scr": {}}
    for u in STRATA16:
        a, b = E[f"{u}_primary"], E["light_primary"]
        sa, sb = r16.scales([a["riskA"]], [a["riskB"]]), r16.scales([b["riskA"]], [b["riskB"]])
        out["pair"][u] = {"OR": float(np.exp(sa["OR"][0] - sb["OR"][0])), "RR": float(np.exp(sa["RR"][0] - sb["RR"][0])), "RD": float(sa["RD"][0] - sb["RD"][0])}
    for u in UNITS16:
        out["scr"][u] = E[f"scr_{u}"]["ratio"]
    return out


def estimate(args):
    P, R = r18.priv(args), r18.res_dir(args)
    t0, start = time.time(), r18.now_iso()
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
        for x in [i for i in ITEMS2 + CHECKS2 if i["pipe"] == p]:
            ref, v = x["stored"](S)[0], x["point"](rj)
            ok = r18.same(v, ref, x["dp"], x["tol"])
            good &= ok
            lines.append(f"{'PASS' if ok else 'FAIL'} | {p} {x['id']} point estimate: {v if v is None else round(v, x['dp'])} vs {round(ref, x['dp'])}")
        est = {"res": rj, "flow": {k: sc.mask_n(v) for k, v in (st.get("flow") or {}).items()}, "n": int(len(st["d"]["A"]))}
        if p == "e1b":
            est["shares_m"] = st["shares_m"]
        json.dump(est, open(pr(P, f"est_{p}.json"), "w", encoding="utf-8"))
        np.savez(pr(P, f"cache_{p}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        out[p] = {"good": bool(good), "lines": lines}
        for ln in lines:
            print(ln, flush=True)
        say(f"{p}: point estimates {'reproduced' if good else 'NOT reproduced'}")
    rp = r16_point(P)
    lines, good = [], True
    for x in [i for i in ITEMS2 if i["pipe"] == "r16s"]:
        ref, v = x["stored"](S)[0], x["point"](rp)
        ok = r18.same(v, ref, 4)
        good &= ok
        lines.append(f"{'PASS' if ok else 'FAIL'} | r16s {x['id']} point estimate from R16's stored estimates: {round(v, 4)} vs {ref}")
    json.dump(rp, open(pr(P, "est_r16s.json"), "w", encoding="utf-8"))
    out["r16s"] = {"good": bool(good), "lines": lines}
    for ln in lines:
        print(ln, flush=True)
    json.dump(out, open(pr(P, "point_checks_b2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    record(P, "estimate", t0, start)


def gate_ok(P, p):
    f = pr(P, "gate_b2.json")
    return os.path.exists(f) and json.load(open(f, encoding="utf-8")).get(p, {}).get("status") == "PASS"


def boot(args):
    P = r18.priv(args)
    p, scheme = args.pipe, args.scheme
    if scheme == "pat" and not gate_ok(P, p):
        raise SystemExit(f"{p}: the reproduction gate has not been passed; no patient-level replicates")
    t0 = time.time()
    est = json.load(open(pr(P, f"est_{p}.json"), encoding="utf-8"))
    st = PIPES[p]["load"](P, est)
    z = np.load(pr(P, f"cache_{p}.npz"))
    cache = {k.replace("__", ":"): z[k] for k in z.files}
    cols, nb = PIPES[p]["cols"], PIPES[p]["nboot"]
    reps = list(range(args.start, min(args.stop, nb)))
    V = np.full((len(reps), len(cols)), np.nan)
    failed = 0
    for i, rep in enumerate(reps):
        try:
            V[i] = np.asarray(PIPES[p]["vec"](st, scheme, rep, cache), float)
        except Exception:
            failed += 1
    np.savez(r18.boot_path(P, p, scheme, args.start, args.stop), reps=np.array(reps), v=V, cols=np.array(cols), failed=failed)
    say(f"{p} {scheme} {args.start}-{args.stop} done, failed {failed} ({time.time() - t0:.0f} s)")


def run(args):
    P = r18.priv(args)
    scheme = args.scheme
    pc = json.load(open(pr(P, "point_checks_b2.json"), encoding="utf-8"))
    pipes = [p for p in ORDER if pc.get(p, {}).get("good") and (scheme == "hosp" or gate_ok(P, p))]
    jobs = [(p, a, min(a + PIPES[p]["chunk"], PIPES[p]["nboot"])) for p in pipes for a in range(0, PIPES[p]["nboot"], PIPES[p]["chunk"])
            if not os.path.exists(r18.boot_path(P, p, scheme, a, min(a + PIPES[p]["chunk"], PIPES[p]["nboot"])))]
    t0, start = time.time(), r18.now_iso()
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


def boots(P, p, scheme):
    if p == "r16s":
        return r16.load_boots(P, "c2e", "main" if scheme == "hosp" else "pat")
    return r18.load_boots(P, p, scheme)


def gate(args):
    P, R = r18.priv(args), r18.res_dir(args)
    t0, start = time.time(), r18.now_iso()
    S = load_stored(R)
    pc = json.load(open(pr(P, "point_checks_b2.json"), encoding="utf-8"))
    out = {}
    for p in ORDER + ["r16s"]:
        lines, good = list(pc.get(p, {}).get("lines", [])), bool(pc.get(p, {}).get("good"))
        H = boots(P, p, "hosp")
        need = 2000 if p == "r16s" else PIPES[p]["nboot"]
        if H is None or H["n"] < need:
            good = False
            lines.append(f"FAIL | {p}: hospital-level replicates missing or incomplete")
        else:
            for x in [i for i in ITEMS2 + CHECKS2 if i["pipe"] == p]:
                _, slo, shi = x["stored"](S)
                q = interval(x["spec"], H)
                tol = x["tol"] if x["kind"] in ("hr",) or x["id"] in ("B2-24", "c-r14b-ic") else None
                ok = q is not None and r18.same(q[0], slo, x["dp"], tol) and r18.same(q[1], shi, x["dp"], tol)
                good &= ok
                nd = x["dp"]
                lines.append(f"{'PASS' if ok else 'FAIL'} | {p} {x['id']} hospital-level interval: " + (f"{q[0]:.{nd}f}-{q[1]:.{nd}f}" if q else "none")
                             + f" vs {round(slo, nd)}-{round(shi, nd)}")
        if p == "r16s":
            Bp = boots(P, p, "pat")
            for u in ("light", "moderate", "deep", "missing"):
                q = interval(("risk", f"{u}_primary", "OR"), Bp)
                ref = S["pat16"][f"{u}_primary"]
                ok = q is not None and r18.same(q[0], ref["OR_lo"]) and r18.same(q[1], ref["OR_hi"])
                good &= ok
                lines.append(f"{'PASS' if ok else 'FAIL'} | r16s stored patient-level replicates, {u} stratum OR: "
                             + (f"{q[0]:.4f}-{q[1]:.4f}" if q else "none") + f" vs {ref['OR_lo']}-{ref['OR_hi']}")
        out[p] = {"status": "PASS" if good else "FAIL", "lines": lines, "hosp_failed_reps": (H or {}).get("failed")}
        for ln in lines:
            if "hospital-level" in ln or ln.startswith("FAIL") or "stored patient-level" in ln:
                print(ln, flush=True)
        say(f"gate {p}: {out[p]['status']}")
    json.dump(out, open(pr(P, "gate_b2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    record(P, "gate", t0, start)


def unrounded_rows(P, S, items, est_of, boots_of, gate_of):
    rows = []
    for x in items:
        p = x["pipe"]
        sv = x["stored"](S)
        r = est_of(p)
        pv = x["point"](r) if r is not None else sv[0]
        qh, qp = interval(x["spec"], boots_of(p, "hosp")), (interval(x["spec"], boots_of(p, "pat")) if gate_of(p) else None)
        nv = x["n"](r) if (x.get("n") and r is not None) else None
        rows.append({"x": x, "pv": pv, "sv": sv, "qh": qh, "qp": qp, "nv": nv})
    return rows


def report(args):
    P, R = r18.priv(args), r18.res_dir(args)
    files = {k: os.path.join(R, f) for k, f in (("j", "r18_batch2.json"), ("m", "r18_batch2.md"), ("c", "r18_cells.json"))}
    if any(os.path.exists(f) for f in files.values()) and not args.overwrite:
        raise SystemExit("results/r18_batch2.* or r18_cells.json exists (use --overwrite)")
    t0, start = time.time(), r18.now_iso()
    S = load_stored(R)
    g2 = json.load(open(pr(P, "gate_b2.json"), encoding="utf-8"))
    g1 = json.load(open(pr(P, "gate.json"), encoding="utf-8"))
    est2 = {p: json.load(open(pr(P, f"est_{p}.json"), encoding="utf-8"))["res"] for p in ORDER}
    est2["r16s"] = json.load(open(pr(P, "est_r16s.json"), encoding="utf-8"))
    est1 = {p: json.load(open(pr(P, f"est_{p}.json"), encoding="utf-8"))["res"] for p in r18.ORDER}
    B2 = {(p, s): boots(P, p, s) for p in ORDER + ["r16s"] for s in ("hosp", "pat")}
    B1 = {(p, s): r18.load_boots(P, p, s) for p in r18.ORDER for s in ("hosp", "pat")}
    checks, good5 = [], True
    for p, spec, ref, tol, lab in PAT_REF2:
        if g2.get(p, {}).get("status") != "PASS":
            continue
        q = interval(spec, B2[(p, "pat")])
        rlo, rhi = ref(S)
        ok = q is not None and r18.same(q[0], rlo, 4, tol) and r18.same(q[1], rhi, 4, tol)
        good5 &= ok
        checks.append(f"{'PASS' if ok else 'FAIL'} | {p} {spec[1]}, patient-level interval vs {lab}: " + (f"{q[0]:.4f}-{q[1]:.4f}" if q else "none")
                      + f" vs {rlo}-{rhi}")
    qn = [interval(("lor", "negctl"), B2[("asd1", "pat")]), interval(("lor", "m0_negctl"), B2[("r7c1", "pat")]),
          interval(("risk", "negctl", "OR"), B2[("c1b", "pat")])]
    ok = all(q is not None for q in qn) and all(r18.same(q[0], round(qn[0][0], 4)) and r18.same(q[1], round(qn[0][1], 4)) for q in qn)
    good5 &= ok
    checks.append(f"{'PASS' if ok else 'FAIL'} | dose comparison negative control, patient-level interval equal in asd.py, r7.py and r16.py: "
                  + "; ".join(f"{q[0]:.4f}-{q[1]:.4f}" if q else "none" for q in qn))
    for ln in checks:
        print(ln, flush=True)
    if not good5:
        raise SystemExit("the patient-level draws do not reproduce the existing patient-level intervals: nothing written (addendum 3.5)")
    rows2 = unrounded_rows(P, S, ITEMS2, lambda p: est2.get(p), lambda p, s: B2[(p, s)], lambda p: g2.get(p, {}).get("status") == "PASS")
    rows1 = unrounded_rows(P, r18.load_stored(R), r18.ITEMS, lambda p: est1.get(p), lambda p, s: B1[(p, s)], lambda p: g1.get(p, {}).get("status") == "PASS")
    out_rows = []
    for w in rows2:
        x, dp = w["x"], w["x"]["dp"]
        rnd = (lambda v: None if v is None else round(float(v), 3 if x["kind"] == "iv" else dp))
        hl, hh = (w["qh"][0], w["qh"][1]) if w["qh"] else (w["sv"][1], w["sv"][2])
        pl, ph = (w["qp"][0], w["qp"][1]) if w["qp"] else (None, None)
        rh, rp = read_cell(x["kind"], hl, hh), (read_cell(x["kind"], pl, ph) if w["qp"] else None)
        out_rows.append({"id": x["id"], "label": x["label"], "script": x["script"], "key": x["key"], "type": x["kind"], "rule": rule_of(x["kind"]),
                         "n": {"PPI_or_low_dose": sc.mask_n(w["nv"][0]), "famotidine_or_high_dose": sc.mask_n(w["nv"][1])} if w["nv"] else None,
                         "estimate": rnd(w["pv"]), "hospital": {"lo": rnd(hl), "hi": rnd(hh), "reported_lo": w["sv"][1], "reported_hi": w["sv"][2],
                                                                "valid": w["qh"][2] if w["qh"] else None, "reading": lab_of(rh)},
                         "patient": None if not w["qp"] else {"lo": rnd(pl), "hi": rnd(ph), "valid": w["qp"][2], "reading": lab_of(rp)},
                         "display": {"estimate": fnum(x["kind"], w["pv"]), "hospital": fspan(x["kind"], hl, hh), "patient": fspan(x["kind"], pl, ph)},
                         "agree": None if (rh == "not_read" or not w["qp"]) else bool(rh == rp)})
    cache_lines, cells_out, unverified = {}, [], []
    for b, rows in ((1, rows1), (2, rows2)):
        for w in rows:
            x = w["x"]
            kind = x["kind"]
            hl, hh = (w["qh"][0], w["qh"][1]) if w["qh"] else (w["sv"][1], w["sv"][2])
            pl, ph = (w["qp"][0], w["qp"][1]) if w["qp"] else (None, None)
            rh, rp = read_cell(kind, hl, hh), (read_cell(kind, pl, ph) if w["qp"] else None)
            locs = []
            for spec in WHERE[x["id"]]:
                loc = locate(spec, cache_lines)
                loc["verified"] = check_print(loc, kind, w["pv"], hl, hh)
                if not loc["verified"]:
                    unverified.append(f"{x['id']} {spec.get('sec') or spec.get('find')}: {loc.get('error', loc.get('printed_text'))}")
                locs.append(loc)
            nd = 2 if x["id"] in ("B1-05", "B2-34") else (3 if kind == "iv" else 4)
            cells_out.append({"id": x["id"], "batch": b, "estimate": x["label"], "script": x["script"], "result_key": x["key"], "type": kind, "rule": rule_of(kind),
                              "printed": locs, "also_drawn_in": FIGURE.get(x["id"]),
                              "hospital": {"text": ftext(kind, w["pv"], hl, hh), "lo": round(hl, nd), "hi": round(hh, nd), "reading": lab_of(rh)},
                              "patient": None if not w["qp"] else {"interval_text": fspan(kind, pl, ph), "lo": round(pl, nd), "hi": round(ph, nd),
                                                                   "valid_samples": w["qp"][2], "reading": lab_of(rp)},
                              "agree": None if (rh == "not_read" or not w["qp"]) else bool(rh == rp)})
    T2 = json.load(open(pr(P, "timing_b2.json"), encoding="utf-8")) if os.path.exists(pr(P, "timing_b2.json")) else {"steps": []}
    T2["steps"].append({"step": "report", "start": start, "end": r18.now_iso(), "seconds": round(time.time() - t0, 1)})
    out = {"plan": r18.PLAN + ", batch 2 (amendment 3)", "gate": {p: g2.get(p, {}).get("status") for p in ORDER + ["r16s"]},
           "checks": {"point_and_hospital": {p: g2.get(p, {}).get("lines", []) for p in ORDER + ["r16s"]}, "patient_vs_existing": checks},
           "boot": {p: {s: {"reps": (B2[(p, s)] or {}).get("n"), "failed": (B2[(p, s)] or {}).get("failed")} for s in ("hosp", "pat")} for p in ORDER + ["r16s"]},
           "batch2": out_rows, "differ": [r["id"] for r in out_rows if r.get("agree") is False], "timing": T2}
    cm = {"plan": r18.PLAN + ", cell map (amendment 3)", "files": {"Supplementary_results_v13.md": r18.sha(os.path.join(MS, AF1)),
                                                                       "Manuscript_full_draft_v18.md": r18.sha(os.path.join(MS, MAIN))},
          "formats": "hospital text as the tables print it (two decimals; risk differences and their differences in percentage points to one decimal; IV risk "
                     "differences to three decimals), rounded once from unrounded numbers; patient-level interval in the same format",
          "near_0.75": NEAR, "unverified_prints": unverified, "differ": [c["id"] for c in cells_out if c["agree"] is False], "estimates": cells_out}
    json.dump(asd.to_json(out), open(files["j"], "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    json.dump(asd.to_json(cm), open(files["c"], "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    open(files["m"], "w", encoding="utf-8", newline="\n").write(md(out, cm))
    say(f"batch 2 written: {len(out['differ'])} batch-2 estimates differ; cell map: {len(cells_out)} estimates, {len(cm['differ'])} differ, "
        f"{len(unverified)} printed texts not verified")


def md(o, cm):
    L = ["# Revision R18 results: second batch (59 estimates) and the cell map of all 98", "",
         f"Plan: {o['plan']}. Method, draws, seeds and reproduction gate as in the first batch (r18.md); readings as in code/r18b.py (addendum section 4 and, as the Methods state, an interval that includes 1 with a lower limit within ≤0.01 of 0.75 is described by its range, "
         "not classified; burden outcomes by the 1-day rule of addendum E2; the instrumental variable is not read). Counts masked: 1–9 as <10, otherwise the nearest 5. Two-decimal values are rounded once from the unrounded numbers.", "",
         "## 1. Reproduction gate", "", "| Pipeline | Reproduction | Hospital level (replicates, failed) | Patient level (replicates, failed) |", "|---|---|---|---|"]
    for p, b in o["boot"].items():
        L.append(f"| {p} | {'passed' if o['gate'].get(p) == 'PASS' else 'not passed'} | {b['hosp']['reps']}, {b['hosp']['failed']} | {b['pat']['reps']}, {b['pat']['failed']} |")
    nf = sum(1 for p in o["checks"]["point_and_hospital"] for ln in o["checks"]["point_and_hospital"][p] if ln.startswith("FAIL"))
    npass = sum(1 for p in o["checks"]["point_and_hospital"] for ln in o["checks"]["point_and_hospital"][p] if ln.startswith("PASS"))
    L += ["", f"Item checks: {npass} passed, {nf} failed (each in r18_batch2.json). r16s are the replicates stored in the private folder of R16 (the hospital-level set reproduces the intervals of R16, the patient-level set the patient-level intervals of R16).", "",
          "Check of the patient-level draws: ", ""] + [f"- {ln}" for ln in o["checks"]["patient_vs_existing"]]
    L += ["", "## 2. Second batch, 59 estimates", "", "| ID | Estimate | Patients | Estimate (hospital-level 95% CI, as reported) | Patient-level 95% CI (usable replicates) | Reading: hospital → patient level | Same |", "|---|---|---|---|---|---|---|"]
    for r in o["batch2"]:
        n = f"{r['n']['PPI_or_low_dose']} / {r['n']['famotidine_or_high_dose']}" if r.get("n") else "—"
        pt = r["patient"]
        L.append(f"| {r['id']} | {r['label']} | {n} | {r['display']['estimate']} ({r['display']['hospital']}) | "
                 + (f"{r['display']['patient']} ({pt['valid']}) | {r['hospital']['reading']} → {pt['reading']} | "
                    + ("—" if r["agree"] is None else ("same" if r["agree"] else "**differ**")) if pt else "not computed | — | —") + " |")
    L += ["", "## 3. Estimates among all 98 whose reading depends on the resampling unit (cell map r18_cells.json)", "",
          "| ID | Estimate | Printed in | Hospital level (as printed) | Patient level | Hospital-level reading | Patient-level reading |", "|---|---|---|---|---|---|---|"]
    for c in cm["estimates"]:
        if c["agree"] is False:
            where = "; ".join(f"{lc.get('where', '')}" + (f" ({lc['row']}, {lc['column']})" if lc.get("row") else "") for lc in c["printed"] if "error" not in lc)
            L.append(f"| {c['id']} | {c['estimate']} | {where} | {c['hospital']['text']} | {c['patient']['interval_text']} | {c['hospital']['reading']} | {c['patient']['reading']} |")
    L += ["", f"Cell check: {len(cm['estimates'])} estimates, {sum(len(c['printed']) for c in cm['estimates'])} printed texts; not checked: "
          + ("no" if not cm["unverified_prints"] else "; ".join(cm["unverified_prints"])) + ".", "",
          "## 4. Running time", "", "| Step | Start | End | Minutes | Note |", "|---|---|---|---|---|"]
    for s in o["timing"]["steps"]:
        extra = f"{s.get('workers', '')} processes, {s.get('chunks', '')} chunks, CPU {s['cpu_seconds'] / 3600:.1f} hours" if "cpu_seconds" in s else ""
        L.append(f"| {s['step']} | {s['start']} | {s['end']} | {s['seconds'] / 60:.1f} | {extra} |")
    return "\n".join(L) + "\n"


def selftest(args):
    sys.excepthook = sys.__excepthook__
    passed, total, lines = 0, 0, []

    def check(name, cond, note=""):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        lines.append(f"{'PASS' if cond else 'FAIL'} | {name}" + (f" ({note})" if note else ""))
        print(lines[-1], flush=True)

    rng = np.random.default_rng(28)
    n = 1200
    d, _ = r16.synth(n, rng, "e", effect=-0.3)
    d["sub"]["cc"] = rng.random(n) < 0.8
    d["sub"]["egfr50"] = rng.random(n) < 0.6
    d["hosp_death"] = (rng.random(n) < 0.12).astype(int)
    st1 = r18.synth_states(d, rng)
    d1 = dict(d, dcfd7=rng.integers(0, 8, n).astype(float), deldays7=rng.integers(0, 4, n).astype(float))
    states = {"asd2": st1["asd"], "r7b": st1["r7"], "r11b": st1["r11"], "c1b": st1["c1"], "e1b": st1["e1"], "r14b": st1["r14"],
              "asd1": {"d": d}, "r7c1": {"dd": st1["r7"]["dd"], "d": st1["r7"]["d"]}, "r7dose": {"dd": {"m0": d}, "d": d}, "e2c1": {"d": d1}}
    for p in ORDER:
        st = states[p]
        res, cache = PIPES[p]["point"](st)
        for scheme in ("hosp", "pat"):
            v1 = np.asarray(PIPES[p]["vec"](st, scheme, 1, cache), float)
            v2 = np.asarray(PIPES[p]["vec"](st, scheme, 1, cache), float)
            check(f"{p} {scheme}: {len(PIPES[p]['cols'])} columns, finite, reproducible", v1.size == len(PIPES[p]["cols"]) and np.isfinite(v1).sum() >= v1.size - 2
                  and np.allclose(v1, v2, equal_nan=True), f"{int(np.isfinite(v1).sum())} finite")
        rj = asd.to_json(res)
        bad = []
        for x in [i for i in ITEMS2 + CHECKS2 if i["pipe"] == p]:
            try:
                v = x["point"](rj)
                ok = v is None or isinstance(v, float)
                if x.get("n"):
                    ok &= all(isinstance(k, int) for k in x["n"](rj))
                c, sp = PIPES[p]["cols"], x["spec"]
                ok &= (sp[1] in c) if sp[0] in ("lor", "raw") else ((sp[1] in c and sp[2] in c) if sp[0] == "ratio" else (sp[1] + ":A" in c))
            except Exception as e:
                ok = False
                bad.append(f"{x['id']} {type(e).__name__}")
            if not ok and not bad:
                bad.append(x["id"])
        check(f"{p}: point getters and interval columns of every item", not bad, "; ".join(bad))
    rr = np.random.default_rng(3)
    B = {"cols": ["a_primary:A", "a_primary:B", "light_primary:A", "light_primary:B"], "v": rr.uniform(0.05, 0.4, (500, 4))}
    q = interval(("pair", "a_primary", "light_primary", "OR"), B)
    ref = r16.paired(B, "a_primary", "light_primary")["OR"]
    check("contrast between strata within replicates equals r16.paired", q is not None and abs(q[0] - ref["lo"]) < 1e-12 and abs(q[1] - ref["hi"]) < 1e-12)
    cases = [("or", 0.60, 0.95, "favours"), ("or", 0.7632, 1.05, "benefit_excluded"), ("or", 0.7354, 1.03, "inconclusive"), ("or", 0.7534, 1.2, "near_0.75"),
             ("or", 0.7400, 1.2, "near_0.75"), ("or", 0.7399, 1.2, "inconclusive"), ("or", 0.745, 0.99, "favours"), ("or", 1.02, 2.2, "opposite"),
             ("nc", 0.9, 1.2, "no_signal"), ("nc", 1.01, 1.4, "signal"), ("ratio", 1.03, 2.26, "excl_above"), ("ratio", 0.97, 1.79, "incl_1"),
             ("ratio", 0.65, 0.98, "excl_below"), ("diff", 0.038, 0.18, "excl0_above"), ("diff", -0.019, 0.21, "incl_0"), ("iv", -0.2, 0.2, "not_read"),
             ("ap", 0.64, 0.96, "fewer"), ("hr", 0.79, 1.03, "unclear")]
    for kind, lo, hi, want in cases:
        check(f"cell reading {kind} ({lo}, {hi}) -> {want}", read_cell(kind, lo, hi) == want)
    check("burden readings follow e2_burden.verdict (and its mirror for ICU days with delirium)",
          read_cell("burden_pos", -0.29, 0.20) == "E2:clinically important benefit excluded (interval includes 0)" and read_cell("burden_pos", 1.1, 2.0) == "E2:clinically important benefit"
          and read_cell("burden_neg", -0.13, 0.22) == "E2:clinically important benefit excluded (interval includes 0)" and read_cell("burden_neg", -2.0, -1.1) == "E2:clinically important benefit")
    check("formats as the tables print them", ftext("or", 0.8650300001, 0.86503, 1.08912) == "0.87 (0.87–1.09)" and fspan("rd", -0.0757, 0.0688) == "−7.6 to 6.9"
          and ftext("iv", -0.0706, -0.1679, 0.2121) == "−0.071 (−0.168 to 0.212)" and ftext("burden_pos", -0.0538, -0.2876, 0.1974) == "−0.05 (−0.29 to 0.20)"
          and ftext("diff", 0.0909, 0.0382, 0.183) == "9.1 (3.8 to 18.3)")
    cache_lines, bad = {}, []
    for i, specs in WHERE.items():
        for spec in specs:
            loc = locate(spec, cache_lines)
            if "error" in loc:
                bad.append(f"{i} {spec.get('sec') or spec.get('find')}: {loc['error']}")
    check(f"all {sum(len(v) for v in WHERE.values())} print locators of the 98 estimates resolve in the current files", not bad, "; ".join(bad))
    check("98 estimates: 39 in batch 1 and 59 in batch 2, each with a print locator", len(r18.ITEMS) == 39 and len(ITEMS2) == 59
          and set(WHERE) == {x["id"] for x in r18.ITEMS + ITEMS2})
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
    x.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    {"selftest": selftest, "extract": extract, "estimate": estimate, "boot": boot, "run": run, "gate": gate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
