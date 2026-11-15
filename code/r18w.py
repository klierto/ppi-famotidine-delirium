#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import r18
import r18b
import r17_mult as rm
from r18 import r16

PLAN = "SAP_addendum_R18.md v1.0, section 7 (amendment 4)"
MS = os.path.join(ROOT, "manuscript_draft")
AF1, MAIN = "Supplementary_results_v13.md", "Manuscript_full_draft_v18.md"
STRATA = ("light", "moderate", "deep", "missing")
NAMES = [f"{u}_primary" for u in STRATA]
PRINTED_WALD = {"OR": "0.0020", "RR": "0.0012", "RD": "0.0073"}
BONF6 = 0.05 / 6
M = "−"
FOLD = [("unless stated; in Tables S13–S23 the eICU-CRD interval from patient-level resampling follows in square brackets;",
         "unless stated; for eICU-CRD the interval from patient-level resampling follows in square brackets;"),
        ("Among the other eICU-CRD analyses, the interpretation", "Among the other eICU-CRD analyses of Table S22, the interpretation")]


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fnum(v, dp=2):
    s = f"{v:.{dp}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", M)


def fspan(scale, lo, hi):
    if scale == "RD":
        return f"{fnum(100 * lo, 1)} to {fnum(100 * hi, 1)}"
    return f"{fnum(lo)}–{fnum(hi)}"


def ftext(scale, x, lo, hi):
    return f"{fnum(100 * x, 1) if scale == 'RD' else fnum(x)} ({fspan(scale, lo, hi)})"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def nums(s):
    return re.findall(r"[−-]?\d+\.\d+", s)


def wald_reading(p):
    if p >= 0.05:
        return "no evidence that the stratum estimates differ (P >= 0.05)"
    return "nominally different (0.0083 <= P < 0.05)" if p >= BONF6 else "different after correction for six dimensions (P < 0.0083)"


def cell_places(cm):
    out = []
    for e in cm["estimates"]:
        for p in e["printed"]:
            if p["file"] == AF1 and not p["point_only"]:
                out.append({"context": p["context"], "printed_text": p["printed_text"], "column": p.get("column"), "line": p["line"],
                            "interval_text": e["patient"]["interval_text"], "id": e["id"], "where": p["where"], "hospital": e["hospital"]["text"]})
    return out


def unbracket(s, places):
    by = {}
    for p in places:
        by.setdefault(p["context"], []).append((p["printed_text"], p["interval_text"]))
    for ctx, ins in by.items():
        done = False
        for c in (ctx, re.sub(r"\| {2,}\|", "| |", ctx)):
            cands = [c]
            for txt, pat in ins:
                nxt = []
                for cc in cands:
                    i = cc.find(txt)
                    while i >= 0:
                        nxt.append(cc[:i + len(txt)] + f" [{pat}]" + cc[i + len(txt):])
                        i = cc.find(txt, i + 1)
                cands = nxt
            hits = [cc for cc in set(cands) if s.count(cc) == 1]
            if len(hits) == 1:
                s, done = s.replace(hits[0], c), True
                break
        if not done:
            return None
    return s


def af1_2214(R):
    cm = json.load(open(os.path.join(R, "r18_cells.json"), encoding="utf-8"))
    want = cm["files"][AF1]
    s = open(os.path.join(MS, AF1), encoding="utf-8").read()
    if sha_text(s) == want:
        return s, cm, "the file as it stands (SHA-256 equal to the 22:14 version)"
    t = unbracket(s, cell_places(cm))
    if t is None:
        raise SystemExit("Additional file 1 is neither the 22:14 version nor that version with the R18 brackets: stop")
    for old, new in FOLD:
        if t.count(new) == 1:
            t = t.replace(new, old)
    if sha_text(t) != want:
        raise SystemExit("the recovered Additional file 1 does not match the SHA-256 of the 22:14 version: stop")
    return t, cm, "recovered from the current file by removing the R18 brackets and the two sentences changed with them; SHA-256 equal to the 22:14 version"


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(‘“])")


def sentence_spans(line):
    spans, a = [], 0
    for m in SENT.finditer(line):
        spans.append((a, m.start()))
        a = m.end()
    spans.append((a, len(line)))
    return spans


def sentence_at(line, pos):
    for a, b in sentence_spans(line):
        if a <= pos < b:
            return a, b
    raise ValueError("position between sentences")


def quote(text, anchor, name):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(f"{name}: '{anchor[:60]}' found {n} times: stop")
    i = text.index(anchor)
    a0 = text.rfind("\n", 0, i) + 1
    b0 = text.find("\n", i)
    line = text[a0:len(text) if b0 < 0 else b0]
    lno = text.count("\n", 0, i) + 1
    if line.startswith("|"):
        return {"file": name, "line": lno, "text": line}
    a, b = sentence_at(line, i - a0)
    while i - a0 + len(anchor) > b:
        b = sentence_at(line, b + 1)[1] if b + 1 < len(line) else len(line)
    return {"file": name, "line": lno, "text": line[a:b]}


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def header_of(lines, i):
    j = i
    while j > 0 and not lines[j].startswith("|---"):
        j -= 1
    return cells(lines[j - 1]), j


def heading_of(lines, i):
    for j in range(i, -1, -1):
        if lines[j].startswith("#"):
            return lines[j].lstrip("#").strip()
    return ""


def row_label(lines, i, nlab):
    hdr, j = header_of(lines, i)
    out = []
    for k in range(nlab):
        v = ""
        for r in range(i, j, -1):
            c = cells(lines[r])
            if k < len(c) and c[k]:
                v = c[k]
                break
        out.append(v)
    return ", ".join(x for x in out if x)


def simulate(text, places):
    by = {}
    for p in places:
        by.setdefault(p["context"], []).append(p)
    for ctx, ins in by.items():
        if text.count(ctx) != 1:
            ctx = re.sub(r"\| {2,}\|", "| |", ctx)
        if text.count(ctx) != 1:
            raise AssertionError(f"context found {text.count(ctx)} times: {ctx[:80]}")
        new = ctx
        for p in ins:
            txt, pat = p["printed_text"], p["interval_text"]
            if new.count(txt) == 1:
                new = new.replace(txt, f"{txt} [{pat}]")
            else:
                pre = text[:text.index(ctx)].rstrip("\n").split("\n")
                j = max(k for k, ln in enumerate(pre) if ln.startswith("|---"))
                k = cells(pre[j - 1]).index(p["column"]) + 1
                cs = new.split("|")
                if cs[k].strip() != txt:
                    raise AssertionError(f"cell under {p['column']} is not {txt}")
                cs[k] = cs[k].replace(txt, f"{txt} [{pat}]")
                new = "|".join(cs)
        text = text.replace(ctx, new)
    return text


def wald_part(P, R):
    J16 = json.load(open(os.path.join(R, "r16.json"), encoding="utf-8"))
    E = json.load(open(os.path.join(P, "r16_est_c2e.json"), encoding="utf-8"))
    Bh, Bp = r16.load_boots(P, "c2e", "main"), r16.load_boots(P, "c2e", "pat")
    if Bh is None or Bp is None:
        raise SystemExit("R16's stored replicates are missing: stop")
    pts = [E["res"][nm] for nm in NAMES]
    wh, wp = r16.wald_strata(pts, Bh, NAMES), r16.wald_strata(pts, Bp, NAMES)
    st = J16["interaction"]["e"]["wald"]
    lines, good = [], True
    for k in ("OR", "RR", "RD"):
        h, s = wh[k], st[k]
        ok = (h is not None and abs(h["Q"] - s["Q"]) < 1e-4 and abs(h["p"] - s["p"]) / s["p"] < 0.005
              and f"{h['p']:.4f}" == PRINTED_WALD[k] and h["reps"] == s["reps"])
        good &= ok
        lines.append(f"{'PASS' if ok else 'FAIL'} | eICU-CRD Wald test, {k}, hospital-level samples: Q {h['Q']:.4f} vs stored {s['Q']}; "
                     f"P {h['p']:.5f} vs stored {s['p']} (printed {PRINTED_WALD[k]}); {h['reps']} samples")
    pat16 = J16["db"]["e"]["est_pat"]
    for nm in NAMES:
        q = r16.risk_entry(E["res"][nm], Bp, nm)
        ok = r18.same(q.get("OR_lo"), pat16[nm]["OR_lo"]) and r18.same(q.get("OR_hi"), pat16[nm]["OR_hi"])
        good &= ok
        lines.append(f"{'PASS' if ok else 'FAIL'} | R16 stored patient-level samples, {nm} OR interval: {q.get('OR_lo', 0):.4f}-{q.get('OR_hi', 0):.4f}"
                     f" vs reported {pat16[nm]['OR_lo']}-{pat16[nm]['OR_hi']}")
    for ln in lines:
        print(ln, flush=True)
    if not good:
        raise SystemExit("reproduction gate 7.2.1 failed: nothing written")
    tests = []
    for k in ("OR", "RR", "RD"):
        h, p = wh[k], wp[k]
        tests.append({"scale": k, "df": h["df"],
                      "hospital": {"Q": round(h["Q"], 4), "P": float(f"{h['p']:.3g}"), "P_4dp": f"{h['p']:.4f}", "printed": PRINTED_WALD[k],
                                   "samples": h["reps"], "reading": wald_reading(h["p"])},
                      "patient": {"Q": round(p["Q"], 4), "P": float(f"{p['p']:.3g}"), "P_4dp": f"{p['p']:.4f}", "samples": p["reps"],
                                  "reading": wald_reading(p["p"])},
                      "conclusion_at_0.05_differs": (h["p"] < 0.05) != (p["p"] < 0.05),
                      "conclusion_at_0.0083_differs": (h["p"] < BONF6) != (p["p"] < BONF6)})
    return {"tests": tests, "gate": lines}, {"E": E, "Bh": Bh, "Bp": Bp, "J16": J16}


PAT_SRC = {("rass", u): ("r16", f"{u}_primary") for u in STRATA}
PAT_SRC.update({("base", "yes"): ("r18", "B2-09"), ("base", "no"): ("r16", "bnone_primary"), ("surg", "yes"): ("r18", "B2-12"),
                ("surg", "no"): ("r18", "B2-11"), ("vent", "yes"): ("r18", "B1-30"), ("vent", "no"): ("r18", "B1-29"),
                ("hosp", "high"): ("r18", "B2-14"), ("hosp", "mid"): ("r18", "B2-15"), ("hosp", "low"): ("r18", "B2-16")})
LABEL = {"rass": "Sedation at time zero", "base": "Delirium assessment in the 24 h before time zero", "surg": "Surgical service / admission from the operating room",
         "vent": "Invasive ventilation", "era": "Calendar period", "hosp": "Hospital's screening intensity"}


def r18_item(i):
    for x in r18.ITEMS:
        if x["id"] == i:
            return x
    for x in r18b.ITEMS2:
        if x["id"] == i:
            return x
    raise KeyError(i)


def r18_interval(P, i, scheme):
    x = r18_item(i)
    return r18b.interval(x["spec"], r18b.boots(P, x["pipe"], scheme))


def r17_rows(E1, J16):
    rows = []
    for db, factors in rm.FAMILY.items():
        for factor, levels in factors:
            for level in levels:
                s = E1["db"][db]["strata"][factor][level]
                lo, hi = s["CI"]
                se = rm.se_from_ci(lo, hi)
                z = math.log(s["OR"]) / se
                row = {"db": db, "factor": factor, "level": level, "OR": s["OR"], "lo": lo, "hi": hi, "se": se, "z": z, "p": rm.p_two(z)}
                if factor == "rass":
                    r = J16["db"][db]["est"][rm.R16KEY[level]]
                    row["se_boot"] = r["se_logOR"]
                    row["p_boot"] = rm.p_two(math.log(s["OR"]) / r["se_logOR"])
                    if db == "e":
                        rp = J16["db"]["e"]["est_pat"][rm.R16KEY[level]]
                        row["se_pat"] = rm.se_from_ci(rp["OR_lo"], rp["OR_hi"])
                        row["p_pat"] = rm.p_two(math.log(s["OR"]) / row["se_pat"])
                rows.append(row)
    p_main = [r["p"] for r in rows]
    p_c1 = [r.get("p_boot", r["p"]) for r in rows]
    p_c2 = [r.get("p_pat", r["p"]) for r in rows]
    for r, a, b, c1, c2 in zip(rows, rm.holm(p_main), rm.bonferroni(p_main), rm.holm(p_c1), rm.holm(p_c2)):
        r["p_holm"], r["p_bonf"], r["p_holm_check1"], r["p_holm_check2"] = a, b, c1, c2
    return rows


def s24_part(P, R, D, cm):
    E1 = json.load(open(os.path.join(R, "ppi_explore_E1.json"), encoding="utf-8"))
    J17 = json.load(open(os.path.join(R, "r17_mult.json"), encoding="utf-8"))
    J16, E, Bh, Bp = D["J16"], D["E"], D["Bh"], D["Bp"]
    rows = r17_rows(E1, J16)
    lines, good = [], len(rows) == len(J17["rows"]) == 25
    for r, j in zip(rows, J17["rows"]):
        keys = ["p", "p_holm", "p_bonf", "p_holm_check1", "p_holm_check2"] + [k for k in ("p_boot", "p_pat") if k in j]
        good &= (r["db"], r["factor"], r["level"]) == (j["db"], j["factor"], j["level"]) and all(abs(r[k] - j[k]) < 1e-12 for k in keys)
    lines.append(f"{'PASS' if good else 'FAIL'} | R17 recomputed: 25 rows; P, Holm, Bonferroni and both checks equal to results/r17_mult.json")
    cmi = {e["id"]: e for e in cm["estimates"]}
    pat = {}
    for (factor, level), (src, key) in PAT_SRC.items():
        s = E1["db"]["e"]["strata"][factor][level]
        if src == "r16":
            qh, qp = r16.risk_entry(E["res"][key], Bh, key), r16.risk_entry(E["res"][key], Bp, key)
            okh = abs(E["res"][key]["OR"] - s["OR"]) < 5e-4 and (factor == "rass" or (abs(qh["OR_lo"] - s["CI"][0]) < 5e-4
                                                                                      and abs(qh["OR_hi"] - s["CI"][1]) < 5e-4))
            ref = J16["db"]["e"]["est_pat"][key]
            okp = r18.same(qp["OR_lo"], ref["OR_lo"]) and r18.same(qp["OR_hi"], ref["OR_hi"])
            pat[(factor, level)] = {"lo": qp["OR_lo"], "hi": qp["OR_hi"], "sd": qp["se_logOR"], "valid": qp["boot_ok"],
                                    "source": f"R16 stored patient-level samples, {key}"}
            lines.append(f"{'PASS' if okh and okp else 'FAIL'} | e {factor} {level}: R16 {key} OR {E['res'][key]['OR']:.4f} and hospital-level "
                         f"{qh['OR_lo']:.4f}-{qh['OR_hi']:.4f} vs Table S4 source {s['OR']:.4f} ({s['CI'][0]:.4f}-{s['CI'][1]:.4f}); patient-level "
                         f"{qp['OR_lo']:.4f}-{qp['OR_hi']:.4f} vs r16.json {ref['OR_lo']}-{ref['OR_hi']}")
        else:
            qh, qp = r18_interval(P, key, "hosp"), r18_interval(P, key, "pat")
            ref = cmi[key]["patient"]
            okh = qh is not None and r18.same(qh[0], s["CI"][0]) and r18.same(qh[1], s["CI"][1])
            okp = qp is not None and r18.same(qp[0], ref["lo"]) and r18.same(qp[1], ref["hi"])
            pat[(factor, level)] = {"lo": qp[0], "hi": qp[1], "valid": qp[2], "source": f"R18 {key} patient-level samples"}
            lines.append(f"{'PASS' if okh and okp else 'FAIL'} | e {factor} {level}: R18 {key} hospital-level {qh[0]:.4f}-{qh[1]:.4f} vs "
                         f"{s['CI'][0]:.4f}-{s['CI'][1]:.4f}; patient-level {qp[0]:.4f}-{qp[1]:.4f} vs r18_cells.json {ref['lo']}-{ref['hi']}")
        good &= okh and okp
    for ln in lines:
        print(ln, flush=True)
    if not good:
        raise SystemExit("reproduction gate 7.2.2 failed: nothing written")
    prow = []
    for r in rows:
        x = {"db": r["db"], "factor": r["factor"], "level": r["level"], "OR": r["OR"]}
        if r["db"] == "e":
            q = pat[(r["factor"], r["level"])]
            se = rm.se_from_ci(q["lo"], q["hi"])
            x.update(lo=q["lo"], hi=q["hi"], p=rm.p_two(math.log(r["OR"]) / se), source=q["source"], valid=q["valid"])
            if r["factor"] == "rass":
                x["p_sd"] = rm.p_two(math.log(r["OR"]) / q["sd"])
        else:
            x.update(lo=r["lo"], hi=r["hi"], p=r["p"], source="unchanged (MIMIC-IV, resampled by patient)")
            if r["factor"] == "rass":
                x["p_sd"] = r["p_boot"]
        prow.append(x)
    pp = [x["p"] for x in prow]
    pc1 = [x.get("p_sd", x["p"]) for x in prow]
    for x, a, b, c in zip(prow, rm.holm(pp), rm.bonferroni(pp), rm.holm(pc1)):
        x["p_holm"], x["p_bonf"], x["p_holm_check1"] = a, b, c
    out, changes = [], []
    for r, x in zip(rows, prow):
        h = {"interval": fspan("OR", r["lo"], r["hi"]), "P": fp(r["p"]), "Holm": fp(r["p_holm"]), "Bonferroni": fp(r["p_bonf"]),
             "Holm_check1": fp(r["p_holm_check1"]) if "p_boot" in r else None, "Holm_check2": fp(r["p_holm_check2"]) if "p_pat" in r else None}
        p = {"interval": fspan("OR", x["lo"], x["hi"]), "P": fp(x["p"]), "Holm": fp(x["p_holm"]), "Bonferroni": fp(x["p_bonf"]),
             "Holm_check1": fp(x["p_holm_check1"]) if "p_sd" in x else None, "source": x["source"]}
        flips = [k for k, a, b in (("P", r["p"], x["p"]), ("Holm", r["p_holm"], x["p_holm"]), ("Bonferroni", r["p_bonf"], x["p_bonf"]))
                 if (a < 0.05) != (b < 0.05)]
        if "p_sd" in x and (r["p_holm_check1"] < 0.05) != (x["p_holm_check1"] < 0.05):
            flips.append("Holm_check1")
        row = {"db": "MIMIC-IV" if r["db"] == "m" else "eICU-CRD", "factor": LABEL[r["factor"]], "stratum": r["level"], "OR": fnum(r["OR"]),
               "hospital_level_as_printed": h, "patient_level": p, "changes_at_0.05": flips}
        out.append(row)
        if flips:
            changes.append(f"{row['db']} {r['factor']} {r['level']}: " + ", ".join(f"{k} {h[k]} -> {p[k]}" for k in flips))
    rl = {}
    for db in "me":
        x = next(y for y in prow if y["db"] == db and y["factor"] == "rass" and y["level"] == "light")
        rl["MIMIC-IV" if db == "m" else "eICU-CRD"] = rm.classify([x["p_holm"], x["p_holm_check1"]])
    opp = [f"{x['db']}:{x['factor']}:{x['level']}" for x in prow if x["OR"] > 1 and x["p_holm"] < 0.05]
    return {"gate": lines, "rows": out, "changes_at_0.05": changes, "reading_light_patient_level": rl,
            "opposite_direction_after_holm_patient_level": opp, "family_size": len(prow)}


REP = [
    (208, "1.01 (0.88–1.46)", None, None, "r16", "bnone_primary", "OR"),
    (210, "0.71, 0.63–0.91", None, None, "r16", "light_primary", "OR"),
    (210, "1.08, 0.83–1.56", None, None, "r16", "light_negctl", "OR"),
    (210, "0.92 (0.41–1.87)", None, None, "r16", "moderate_negctl", "OR"),
    (210, "0.78 (0.33–1.19)", None, None, "r16", "deep_negctl", "OR"),
    (210, "0.91 (0.55–1.42)", None, None, "r16", "missing_negctl", "OR"),
    (210, "1.07 (0.89–1.41)", None, None, "r16", "rass3_negctl", "OR"),
    (212, "0.91 (0.82–1.05)", None, None, "r16ic", "ic_all", "HR"),
    (216, "0.75, 0.64–0.95", None, None, "r12", "negctl", "ratio"),
    (216, "0.87, 0.77–1.08", None, None, "r12", "primary", "ratio"),
    (240, "0.97, 0.90–1.13", None, None, "r16", "primary", "OR"),
    (244, "0.97 (0.91–1.12)", None, None, "r16", "primary", "RR"),
    (244, "−0.4 (−1.2 to 1.2)", None, None, "r16", "primary", "RD"),
    (248, "0.99 (0.79–1.43)", None, None, "r16", "bneg_negctl", "OR"),
    (248, "1.10 (0.27–2.43)", None, None, "r16", "buta_primary", "OR"),
    (250, "0.97 (0.88–1.15)", None, None, "r16", "h24_primary", "OR"),
    (250, "0.90 (0.83–1.05)", None, None, "r16", "lm6_all", "OR"),
    (250, "0.98 (0.89–1.19)", None, None, "r16", "lm12_all", "OR"),
    (250, "0.67 (0.59–0.87)", None, None, "r16", "lm6_light", "OR"),
    (250, "0.71 (0.62–0.93)", None, None, "r16", "lm12_light", "OR"),
    (252, "0.96 (0.87–1.12)", None, None, "r16", "sev_primary", "OR"),
    (252, "0.97 (0.90–1.14)", None, None, "r16", "kadj_primary", "OR"),
    (252, "0.98 (0.91–1.17)", None, None, "r16", "kexc_primary", "OR"),
    (252, "0.96 (0.89–1.11)", None, None, "r16", "dx_primary", "OR"),
    (400, "0.97 (0.90–1.13)", "eICU-CRD", "Main analysis", "r16", "primary", "OR"),
    (402, "0.96 (0.86–1.09)", "eICU-CRD", "Delirium within 72 h", "r16", "del72", "OR"),
    (407, "0.79 (0.72–0.95)", "eICU-CRD", "Recorded RASS ≥ −3 at time zero", "r16", "rass3_primary", "OR"),
    (408, "0.97 (0.78–1.41)", "eICU-CRD", "eGFR <60", "r16", "egfr60", "OR"),
    (409, "0.87 (0.77–1.13)", "eICU-CRD", "Age ≥65 years", "r16", "age65", "OR"),
    (417, "0.91 (0.82–1.05)", "eICU-CRD", "Interval-censored hazard ratio, baseline in 3 intervals", "r16ic", "ic_all", "HR"),
    (419, "0.76 (0.66–0.89)", "eICU-CRD", "Interval-censored hazard ratio, recorded RASS", "r16ic", "ic_rass3", "HR"),
    (422, "0.99 (0.86–1.22)", "eICU-CRD", "Main analysis", "r16", "negctl", "OR"),
    (446, "0.71 (0.63–0.91)", "eICU-CRD OR", "RASS at time zero: awake or light", "r16", "light_primary", "OR"),
    (447, "1.31 (1.07–2.20)", "eICU-CRD OR", "moderate", "r16", "moderate_primary", "OR"),
    (448, "1.36 (0.83–2.33)", "eICU-CRD OR", "deep", "r16", "deep_primary", "OR"),
    (449, "1.51 (1.02–2.58)", "eICU-CRD OR", "not recorded", "r16", "missing_primary", "OR"),
    (451, "1.01 (0.88–1.46)", "eICU-CRD OR", "no", "r16", "bnone_primary", "OR"),
    (466, "0.71 (0.63–0.91)", "eICU-CRD delirium", "Awake or light", "r16", "light_primary", "OR"),
    (466, "1.08 (0.83–1.56)", "eICU-CRD negative control", "Awake or light", "r16", "light_negctl", "OR"),
    (467, "1.31 (1.07–2.20)", "eICU-CRD delirium", "Moderate", "r16", "moderate_primary", "OR"),
    (467, "0.92 (0.41–1.87)", "eICU-CRD negative control", "Moderate", "r16", "moderate_negctl", "OR"),
    (468, "1.36 (0.83–2.33)", "eICU-CRD delirium", "Deep", "r16", "deep_primary", "OR"),
    (468, "0.78 (0.33–1.19)", "eICU-CRD negative control", "Deep", "r16", "deep_negctl", "OR"),
    (469, "1.51 (1.02–2.58)", "eICU-CRD delirium", "Not recorded", "r16", "missing_primary", "OR"),
    (469, "0.91 (0.55–1.42)", "eICU-CRD negative control", "Not recorded", "r16", "missing_negctl", "OR"),
    (470, "0.79 (0.72–0.95)", "eICU-CRD delirium", "Recorded RASS ≥ −3", "r16", "rass3_primary", "OR"),
    (470, "1.07 (0.89–1.41)", "eICU-CRD negative control", "Recorded RASS ≥ −3", "r16", "rass3_negctl", "OR"),
    (655, "0.97 (0.90–1.13)", "Main model", "Delirium, 7-day OR", "r16", "primary", "OR"),
    (656, "0.91 (0.82–1.05)", "Main model", "Hazard ratio, interval-censored", "r16ic", "ic_all", "HR"),
    (657, "0.99 (0.86–1.22)", "Main model", "Negative control, 7-day OR", "r16", "negctl", "OR"),
    (718, "0.74 (0.67–0.92)", "RR (95% CI)", "Awake or light (RASS ≥ −1), eICU-CRD", "r16", "light_primary", "RR"),
    (718, "−3.6 (−5.1 to −0.9)", "RD, percentage points (95% CI)", "Awake or light (RASS ≥ −1), eICU-CRD", "r16", "light_primary", "RD"),
    (720, "1.21 (1.05–1.83)", "RR (95% CI)", "Moderate (−3 to −2), eICU-CRD", "r16", "moderate_primary", "RR"),
    (720, "5.5 (1.5 to 14.2)", "RD, percentage points (95% CI)", "Moderate (−3 to −2), eICU-CRD", "r16", "moderate_primary", "RD"),
    (722, "1.21 (0.89–1.76)", "RR (95% CI)", "Deep (≤ −4), eICU-CRD", "r16", "deep_primary", "RR"),
    (722, "7.2 (−4.4 to 18.4)", "RD, percentage points (95% CI)", "Deep (≤ −4), eICU-CRD", "r16", "deep_primary", "RD"),
    (724, "1.48 (1.02–2.47)", "RR (95% CI)", "RASS not recorded, eICU-CRD", "r16", "missing_primary", "RR"),
    (724, "2.3 (0.1 to 5.8)", "RD, percentage points (95% CI)", "RASS not recorded, eICU-CRD", "r16", "missing_primary", "RD"),
    (1041, "0.71 (0.63–0.91)", "OR (95% CI)", "eICU-CRD, Sedation at time zero, Awake or light", "r16", "light_primary", "OR"),
    (1042, "1.31 (1.07–2.20)", "OR (95% CI)", "eICU-CRD, Sedation at time zero, Moderate", "r16", "moderate_primary", "OR"),
    (1043, "1.36 (0.83–2.33)", "OR (95% CI)", "eICU-CRD, Sedation at time zero, Deep", "r16", "deep_primary", "OR"),
    (1044, "1.51 (1.02–2.58)", "OR (95% CI)", "eICU-CRD, Sedation at time zero, RASS not recorded", "r16", "missing_primary", "OR"),
    (1046, "1.01 (0.88–1.46)", "OR (95% CI)", "eICU-CRD, Delirium assessment in the 24 h before time zero, None", "r16", "bnone_primary", "OR"),
    (1055, "1.31, 1.07–2.20", None, None, "r16", "moderate_primary", "OR"),
    (1055, "1.51, 1.02–2.58", None, None, "r16", "missing_primary", "OR"),
]
NLAB = {"Table S14": 2, "Table S24": 3}
FLAG = {1055: "the sentence illustrates the disagreement between the hospital-level percentile interval and the P value derived "
              "from it; the bracket may be left out there"}
EXCLUDED = [
    (216, "0.74–1.10 with the hospital-level bootstrap; 0.74–0.96 with a patient-level bootstrap", "patient-level interval printed beside it"),
    (216, "(0.64–0.97; 0.65–0.83)", "patient-level interval printed beside it (second interval)"),
    (216, "(0.61–1.11; 0.63–0.87)", "patient-level interval printed beside it (second interval)"),
    (246, "with hospital-level resampling (1.07–2.20) but not with patient-level resampling (0.86–1.99)", "explicit: patient-level interval printed"),
    (246, "under both (1.02–2.58 and 1.02–2.23)", "explicit: patient-level interval printed"),
    (248, "0.79 (0.68–0.98) in eICU-CRD, 0.64–0.98 with patient-level resampling", "explicit: patient-level interval printed"),
    (248, "0.74 (0.63–1.03); the eICU-CRD interval included 1 with hospital-level resampling but not with patient-level resampling (0.58–0.96)",
     "explicit: patient-level interval printed"),
    (254, "left the interval at 0.90–1.13", "describes the hospital-level bootstrap with a separate indicator for every drawn copy (Table S22), not a print of the estimate"),
    (605, "| 0.97 | 0.90–1.13 | 0.84–1.11 |", "Table S10: the row has a patient-level column (all nine rows)"),
    (985, "| 0.97 | 0.90–1.13 | 0.84–1.11 |", "Table S22: the row has a patient-level column (rows 985–1001); rows 975–978 compare bootstrap methods"),
]


def r16_values(D, key, scale):
    J16, E, Bp = D["J16"], D["E"], D["Bp"]
    if scale == "HR":
        h = J16["db"]["e"]["ic"][key]
        q = r16.log_entry(E["res"][key].get("HR"), Bp, key)
        ref = J16["db"]["e"]["ic_pat"][key]
        ok = r18.same(q["lo"], ref["lo"]) and r18.same(q["hi"], ref["hi"])
        return (h["HR"], h["lo"], h["hi"]), (q["lo"], q["hi"]), ok, f"r16.json db.e.ic_pat.{key} (HR)"
    h = J16["db"]["e"]["est"][key]
    q = r16.risk_entry(E["res"][key], Bp, key)
    ref = J16["db"]["e"]["est_pat"][key]
    ok = r18.same(q[f"{scale}_lo"], ref[f"{scale}_lo"]) and r18.same(q[f"{scale}_hi"], ref[f"{scale}_hi"])
    return (h[scale], h[f"{scale}_lo"], h[f"{scale}_hi"]), (q[f"{scale}_lo"], q[f"{scale}_hi"]), ok, f"r16.json db.e.est_pat.{key} ({scale})"


def repeat_map(D, R, text, cm):
    L = text.split("\n")
    J12 = json.load(open(os.path.join(R, "r12.json"), encoding="utf-8"))
    cmp_ = cell_places(cm)
    by_line = {}
    for p in cmp_:
        by_line.setdefault(p["line"], []).append(p)
    good, lines, places = True, [], []
    t10 = {"primary": 607, "negctl": 613}
    for ln, txt, col, rowx, src, key, scale in REP:
        line = L[ln - 1]
        if src == "r12":
            h = J12["ratio"][key + "_hosp"]
            hv, pv, ok = (h["ratio"], h["lo"], h["hi"]), (J12["ratio"][key + "_pat"]["lo"], J12["ratio"][key + "_pat"]["hi"]), True
            c10 = cells(L[t10[key] - 1])
            ok = "without vs with" in c10[1] and c10[4] == fspan("OR", *pv)
            skey = f"r12.json ratio.{key}_pat (ratio of ORs, without vs with hospital indicators; R12 samples deleted: 4-decimal values, equal to Table S10's patient-level column)"
        else:
            hv, pv, ok, skey = r16_values(D, key, "HR" if src == "r16ic" else scale)
        sc = "RD" if scale == "RD" else "OR"
        want = [fnum(100 * v, 1) if sc == "RD" else fnum(v) for v in hv]
        ok &= nums(txt) == want
        if col is None:
            ok &= line.count(txt) == 1
            pos = line.index(txt) if txt in line else 0
            ctx = next((p["context"] for p in by_line.get(ln, []) if p["context"] in line
                        and line.index(p["context"]) <= pos and pos + len(txt) <= line.index(p["context"]) + len(p["context"])), None)
            if ctx is None:
                a, b = sentence_at(line, pos)
                ctx = line[a:b]
                for p in by_line.get(ln, []):
                    if p["context"] in line:
                        c0 = line.index(p["context"])
                        ok &= b <= c0 or a >= c0 + len(p["context"])
            where, row = heading_of(L, ln - 1), None
        else:
            hdr, _ = header_of(L, ln - 1)
            cs = cells(line)
            k = hdr.index(col) if col in hdr else -1
            ok &= k >= 0 and cs[k].startswith(txt)
            where = heading_of(L, ln - 1).split(".")[0]
            row = row_label(L, ln - 1, NLAB.get(where, 1))
            ok &= row.startswith(rowx)
            ctx = line
        ok &= text.count(ctx) == 1 and ctx.count(txt) >= 1
        good &= ok
        e = {"file": AF1, "where": where, "row": row, "column": col, "line": ln, "printed_text": txt, "context": ctx, "point_only": False,
             "interval_text": fspan(sc, *pv), "source_key": skey, "scale": "HR" if src == "r16ic" else scale,
             "hospital_text": ftext(sc, *hv), "patient_limits": [round(v, 4) for v in pv]}
        if ln in FLAG:
            e["note"] = FLAG[ln]
        places.append(e)
        lines.append(f"{'PASS' if ok else 'FAIL'} | line {ln} {where[:22]} | {txt} -> [{e['interval_text']}] | {skey.split(' (')[0]}")
    for ln, txt, why in EXCLUDED:
        ok = txt in L[ln - 1]
        good &= ok
        if not ok:
            lines.append(f"FAIL | excluded place not found on line {ln}: {txt[:50]}")
    s = text
    for old, new in FOLD:
        s = s.replace(old, new)
    try:
        out = simulate(s, cmp_ + places)
        O = out.split("\n")
        for e in places:
            tgt = f"{e['printed_text']} [{e['interval_text']}]"
            if e["column"] is None:
                okp = O[e["line"] - 1].count(tgt) == 1
            else:
                hdr, _ = header_of(O, e["line"] - 1)
                okp = cells(O[e["line"] - 1])[hdr.index(e["column"])].startswith(tgt)
            good &= okp
            if not okp:
                lines.append(f"FAIL | simulated bracket not after the interval: line {e['line']} {e['printed_text']}")
        added = len(out) - len(s)
        want_add = sum(len(f" [{p['interval_text']}]") for p in cmp_ + places)
        good &= added == want_add
        sim = f"{len(cmp_)} cell-map and {len(places)} repeat brackets inserted by the insertion rule; each repeat bracket directly follows its interval"
    except AssertionError as err:
        good, sim = False, f"simulation failed: {err}"
    lines.append(("PASS | " if good else "FAIL | ") + sim)
    chk, nok = [], 0
    for p in cmp_:
        m = re.match(r"^\S+ \((.+)\)$", p["hospital"])
        core = m.group(1) if m else None
        txt = p["printed_text"]
        i = txt.find(core) if core else -1
        if i < 0:
            chk.append({"id": p["id"], "line": p["line"], "printed_text": txt, "status": "interval not found in the printed text"})
            continue
        end = i + len(core)
        paren = txt[i - 1:i] == "(" and txt[end:end + 1] == ")"
        if end == len(txt) or (paren and end + 1 == len(txt)):
            nok += 1
            continue
        fixed = txt[:end + (1 if paren else 0)]
        status = ("after the closing parenthesis of the phrase" if txt[end:] == ")" else f"after the words that follow the interval: '{txt[end + (1 if paren else 0):].strip()}'")
        chk.append({"id": p["id"], "line": p["line"], "where": p["where"], "printed_text": txt, "bracket_lands": status,
                    "printed_text_ending_at_interval": fixed, "unique_in_context": p["context"].count(fixed) == 1})
    for ln in lines:
        print(ln, flush=True)
    if not good:
        raise SystemExit("the repeat map failed a check (7.4): nothing written")
    return {"places": places, "lines": lines, "simulation": sim, "cell_map_check": chk, "cell_map_ok": nok,
            "excluded": [{"line": ln, "printed": txt, "reason": why} for ln, txt, why in EXCLUDED]}


QUOTES = {"wald": [("AF1", "0.0020, 0.0012 and 0.0073 in eICU-CRD)"), ("AF1", "| Test of equal estimates in the four strata, P |"),
                   ("AF1", "|  | eICU-CRD | 0.0020 | 0.0012 | 0.0073 |"), ("AF1", "Test: Wald test of equality of the four stratum estimates"),
                   ("MAIN", "Estimates differed by sedation and ventilation at time zero"), ("MAIN", "and 0.0020, 0.0012 and 0.0073 in eICU-CRD"),
                   ("MAIN", "For sedation, a Wald test compared the four strata")],
          "holm": [("AF1", "After Holm correction across the 25 stratum estimates of the six stratifying factors (Table S24)"),
                   ("AF1", "The last two columns repeat the Holm correction"), ("AF1", "when every applicable adjusted P value was below 0.05"),
                   ("AF1", "The derived P values assume a symmetric distribution"),
                   ("MAIN", "After Holm correction across the 25 stratum estimates of the six factors"),
                   ("MAIN", "Holm P: P value of a two-sided test of an OR of 1")]}


def quotes(texts):
    out = {}
    for test, anchors in QUOTES.items():
        out[test] = []
        for f, a in anchors:
            q = quote(texts[f][0], a, texts[f][1])
            if q not in out[test]:
                out[test].append(q)
    return out


def md(o):
    L = [f"# R18 amendment 4: patient-level versions of the test statistics (eICU-CRD)", "",
         f"Basis: {o['plan']}. Text: {o['files'][AF1]['how']}.", "",
         "## 1. Wald test of equal estimates in the four sedation strata (3 degrees of freedom; OR and RR on the log scale; covariance from 2,000 replicates; point estimates unchanged)", "",
         "| Scale | Hospital-level P (printed) | Hospital-level Q | Patient-level P | Patient-level Q | Conclusion changes at 0.05 | Conclusion changes at 0.0083 |", "|---|---|---|---|---|---|---|"]
    for t in o["wald"]["tests"]:
        h, p = t["hospital"], t["patient"]
        L.append(f"| {t['scale']} | {h['P_4dp']} ({h['printed']}) | {h['Q']:.2f} | {p['P_4dp']} | {p['Q']:.2f} | "
                 f"{'yes' if t['conclusion_at_0.05_differs'] else 'no'} | {'yes' if t['conclusion_at_0.0083_differs'] else 'no'} |")
    L += ["", "Reproduction: " + "; ".join(x.split(" | ", 1)[1] for x in o["wald"]["gate"][:3]), "",
          "## 2. Table S24 (25 strata; 13 eICU-CRD strata with patient-level intervals, MIMIC-IV unchanged)", "",
          "| Database | Stratifying factor | Stratum | OR | Hospital-level interval | Patient-level interval | P (hospital → patient) | Holm (hospital → patient) | Bonferroni (hospital → patient) | Changes at 0.05 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in o["s24"]["rows"]:
        h, p = r["hospital_level_as_printed"], r["patient_level"]
        L.append(f"| {r['db']} | {r['factor']} | {r['stratum']} | {r['OR']} | {h['interval']} | {p['interval']} | {h['P']} → {p['P']} | "
                 f"{h['Holm']} → {p['Holm']} | {h['Bonferroni']} → {p['Bonferroni']} | {', '.join(r['changes_at_0.05']) or '—'} |")
    L += ["", f"Least sedated stratum (patient-level version, reading rule of R17): {o['s24']['reading_light_patient_level']}; strata remaining after Holm with OR > 1: "
          f"{o['s24']['opposite_direction_after_holm_patient_level'] or 'no'}.", "",
          "## 3. Sentences that state these conclusions (verbatim)", ""]
    for test, qs in o["sentences"].items():
        L.append(f"### {'Wald test' if test == 'wald' else 'Holm correction (Table S24, Figure 4)'}")
        L.append("")
        for q in qs:
            L.append(f"- {q['file']}, line {q['line']}: {q['text']}")
        L.append("")
    L += ["## 4. Places where an interval is repeated", "", f"results/r18_cells_repeat.json: {len(o['repeat_places'])} places; {o['repeat_simulation']}.", ""]
    return "\n".join(L) + "\n"


def run(args):
    P, R = r18.priv(args), r18.res_dir(args)
    outs = {k: os.path.join(R, f) for k, f in (("j", "r18_wald.json"), ("m", "r18_wald.md"), ("c", "r18_cells_repeat.json"))}
    if any(os.path.exists(f) for f in outs.values()) and not args.overwrite:
        raise SystemExit("results/r18_wald.* or r18_cells_repeat.json exists (use --overwrite)")
    text, cm, how = af1_2214(R)
    main = open(os.path.join(MS, MAIN), encoding="utf-8").read()
    W, D = wald_part(P, R)
    S = s24_part(P, R, D, cm)
    C = repeat_map(D, R, text, cm)
    Q = quotes({"AF1": (text, AF1 + " (22:14)"), "MAIN": (main, MAIN)})
    files = {AF1: {"sha256": sha_text(text), "how": how}, MAIN: {"sha256": sha_text(main), "same_as_cell_map": sha_text(main) == cm["files"][MAIN]}}
    o = {"plan": PLAN, "files": files,
         "wald": dict(W, method="r16.wald_strata: Wald test that the four sedation-stratum estimates are equal (contrasts with the awake or lightly "
                                "sedated stratum), log scale for OR and RR, covariance of the 2,000 bootstrap samples with all strata estimated, "
                                "3 df; points unchanged; hospital level = R16's stored hospital-level samples, patient level = R16's stored "
                                "patient-level samples (seed 20261013 + sample number)"),
         "s24": dict(S, method="r17_mult.py: two-sided Wald test of each log OR with the standard error from the 95% percentile interval, Holm over "
                               "the 25 estimates (Bonferroni for reference); patient level: the 13 eICU-CRD strata from patient-level resampling "
                               "(unrounded limits), MIMIC-IV unchanged; check 1 with the standard deviation of the patient-level log ORs"),
         "out_of_scope": "pooled estimates, I2 and the ratios and differences between the databases use the eICU-CRD hospital-level standard error "
                         "but are not single-database tests and print no P value; not recomputed",
         "sentences": Q, "repeat_places": C["places"], "repeat_simulation": C["simulation"]}
    rep = {"plan": PLAN + ": lookup only, no new estimate",
           "files": {AF1: {"sha256": sha_text(text), "how": how}},
           "formats": "printed_text as printed (it ends with the hospital-level interval, so the bracket goes right after it); interval_text in "
                      "the format of the place, rounded once from unrounded values (two decimals; risk differences in percentage points to "
                      "one decimal)",
           "sources": {"r16": "R16 est_pat / ic_pat: unrounded limits from R16's stored patient-level samples with R16's own functions "
                              "(risk_entry, log_entry), equal to results/r16.json at 4 decimals",
                       "r12": "R12 ratios of ORs without vs with hospital indicators: results/r12.json (R12's samples were deleted), equal to "
                              "Table S10's patient-level column"},
           "contexts": "table rows: the whole row; running text: the sentence, or the context r18_cells.json already uses when the place lies "
                       "inside it (so both are applied together); each context occurs once in the 22:14 text",
           "places": C["places"], "excluded": C["excluded"], "simulation": C["simulation"],
           "cell_map_insertion_check": {"rule": "the build appends the bracket after printed_text; it belongs right after the interval",
                                        "places_checked": len(cell_places(cm)), "bracket_after_interval": C["cell_map_ok"],
                                        "elsewhere": C["cell_map_check"]}}
    json.dump(o, open(outs["j"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(outs["m"], "w", encoding="utf-8", newline="\n").write(md(o))
    json.dump(rep, open(outs["c"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for t in W["tests"]:
        print(f"Wald {t['scale']}: hospital P {t['hospital']['P_4dp']} | patient P {t['patient']['P_4dp']} | differs at 0.05: {t['conclusion_at_0.05_differs']}")
    print("Table S24 changes at 0.05:", S["changes_at_0.05"] or "none")
    print("repeat places:", len(C["places"]), "| cell-map brackets not right after the interval:", len(C["cell_map_check"]))


def selftest(args):
    rng = np.random.default_rng(7)
    n = 2000
    risks = {"light_primary": (0.10, 0.14), "moderate_primary": (0.30, 0.25), "deep_primary": (0.40, 0.33), "missing_primary": (0.07, 0.05)}
    cols, V = [], []
    for nm, (a, b) in risks.items():
        cols += [nm + ":A", nm + ":B"]
        V += [np.clip(a + rng.normal(0, 0.005, n), 0.01, 0.99), np.clip(b + rng.normal(0, 0.005, n), 0.01, 0.99)]
    B = {"cols": cols, "v": np.column_stack(V)}
    pts = [{"OR": (a / (1 - a)) / (b / (1 - b)), "RR": a / b, "RD": a - b} for a, b in risks.values()]
    w = r16.wald_strata(pts, B, NAMES)
    for k in ("OR", "RR", "RD"):
        th = np.array([math.log(p[k]) if k != "RD" else p[k] for p in pts])
        reps = np.column_stack([r16.scales(B["v"][:, 2 * j], B["v"][:, 2 * j + 1])[k] for j in range(4)])
        Dm = reps[:, 1:] - reps[:, [0]]
        d = th[1:] - th[0]
        Q = float(d @ np.linalg.solve(np.cov(Dm, rowvar=False), d))
        assert abs(Q - w[k]["Q"]) < 1e-8 * max(1, Q), (k, Q, w[k]["Q"])
        assert w[k]["p"] < 0.001 and w[k]["df"] == 3
    same = [{"OR": pts[0]["OR"], "RR": pts[0]["RR"], "RD": pts[0]["RD"]}] * 4
    w0 = r16.wald_strata(same, B, NAMES)
    assert all(abs(w0[k]["Q"]) < 1e-9 and w0[k]["p"] > 0.999 for k in ("OR", "RR", "RD"))
    assert wald_reading(0.2).startswith("no evidence") and wald_reading(0.02).startswith("nominally") and wald_reading(0.001).startswith("different")
    rm.selftest()
    assert fnum(-0.0001) == "0.00" and fnum(-0.04, 1) == "0.0" and fspan("RD", -0.0562, -0.0161) == "−5.6 to −1.6"
    assert ftext("OR", 0.7104, 0.6339, 0.9082) == "0.71 (0.63–0.91)" and ftext("RD", -0.0036, -0.0118, 0.0124) == "−0.4 (−1.2 to 1.2)"
    assert fp(0.0004) == "<0.001" and fp(0.0123) == "0.012" and nums("−0.4 (−1.2 to 1.2)") == ["−0.4", "−1.2", "1.2"]
    para = "Lead. The estimate was 0.97 (0.90–1.13) in all and 0.85 (0.71–0.98) in some (ratio 1.19, 1.00–1.94). Next one 1.2 here."
    assert para[slice(*sentence_at(para, para.index("0.85")))].startswith("The estimate was") and len(sentence_spans(para)) == 3
    t = "\n".join(["# Table S9. Test", "", "| Row | Main model | Restricted model |", "|---|---|---|", "| A | 0.99 (0.86–1.22) | 0.99 (0.86–1.22) |",
                   "", "Text with 0.85 (0.71–0.98) in patients with any assessment and 1.01 (0.88–1.46) in those with none. Next."])
    row = "| A | 0.99 (0.86–1.22) | 0.99 (0.86–1.22) |"
    sent = "Text with 0.85 (0.71–0.98) in patients with any assessment and 1.01 (0.88–1.46) in those with none."
    pl = [{"context": row, "printed_text": "0.99 (0.86–1.22)", "column": "Restricted model", "interval_text": "0.83–1.16"},
          {"context": sent, "printed_text": "0.85 (0.71–0.98) in patients with any assessment", "column": None, "interval_text": "0.71–1.02"},
          {"context": row, "printed_text": "0.99 (0.86–1.22)", "column": "Main model", "interval_text": "0.84–1.17"},
          {"context": sent, "printed_text": "1.01 (0.88–1.46)", "column": None, "interval_text": "0.82–1.25"}]
    out = simulate(t, pl)
    assert "| A | 0.99 (0.86–1.22) [0.84–1.17] | 0.99 (0.86–1.22) [0.83–1.16] |" in out, out
    assert "any assessment [0.71–1.02] and 1.01 (0.88–1.46) [0.82–1.25] in those" in out
    assert unbracket(out, pl) == t
    tl = t.split("\n")
    assert header_of(tl, 4)[0] == ["Row", "Main model", "Restricted model"] and row_label(tl, 4, 1) == "A" and heading_of(tl, 4) == "Table S9. Test"
    assert quote(t, "1.01 (0.88–1.46)", "x")["text"] == sent and quote(t, "| A |", "x")["text"] == row
    msg = "self-test passed: Wald test (direct computation, equal strata), Holm and Bonferroni, formats, sentences, insertion rule and its inverse"
    print(msg)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("selftest")
    a.add_argument("--out")
    a = sp.add_parser("run")
    a.add_argument("--private", required=True)
    a.add_argument("--results", required=True)
    a.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    {"selftest": selftest, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
