import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
Z975 = 1.959964
FAMILY = {
    "m": [("rass", ["light", "moderate", "deep", "missing"]), ("base", ["yes", "no"]), ("surg", ["yes", "no"]),
          ("vent", ["yes", "no"]), ("era", ["early", "late"])],
    "e": [("rass", ["light", "moderate", "deep", "missing"]), ("base", ["yes", "no"]), ("surg", ["yes", "no"]),
          ("vent", ["yes", "no"]), ("hosp", ["high", "mid", "low"])],
}
R16KEY = {"light": "light_primary", "moderate": "moderate_primary", "deep": "deep_primary",
          "missing": "missing_primary"}


def p_two(z):
    return math.erfc(abs(z) / math.sqrt(2))


def se_from_ci(lo, hi):
    return (math.log(hi) - math.log(lo)) / (2 * Z975)


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, running = [None] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def bonferroni(ps):
    return [min(1.0, len(ps) * p) for p in ps]


def selftest():
    ps = [0.01, 0.04, 0.03, 0.005]
    got = [round(x, 10) for x in holm(ps)]
    assert got == [0.03, 0.06, 0.06, 0.02], got
    assert bonferroni(ps) == [0.04, 0.16, 0.12, 0.02]
    assert abs(p_two(1.959964) - 0.05) < 1e-6
    print("selftest passed")


def classify(ps):
    ok = [p < 0.05 for p in ps]
    return "remains" if all(ok) else ("not clear" if not any(ok) else "depends")


def main():
    E = json.load(open(os.path.join(RES, "ppi_explore_E1.json"), encoding="utf-8"))
    R = json.load(open(os.path.join(RES, "r16.json"), encoding="utf-8"))
    rows = []
    for db, factors in FAMILY.items():
        for factor, levels in factors:
            for level in levels:
                s = E["db"][db]["strata"][factor][level]
                assert s.get("CI") and len(s["CI"]) == 2 and None not in s["CI"], (db, factor, level, "no interval")
                lo, hi = s["CI"]
                se = se_from_ci(lo, hi)
                z = math.log(s["OR"]) / se
                row = {"db": db, "factor": factor, "level": level, "OR": s["OR"], "lo": lo, "hi": hi,
                       "se": se, "z": z, "p": p_two(z)}
                if factor == "rass":
                    r = R["db"][db]["est"][R16KEY[level]]
                    assert abs(r["OR"] - s["OR"]) < 5e-4, (db, level, r["OR"], s["OR"])
                    row["se_boot"] = r["se_logOR"]
                    row["p_boot"] = p_two(math.log(s["OR"]) / r["se_logOR"])
                    if db == "e":
                        rp = R["db"]["e"]["est_pat"][R16KEY[level]]
                        assert abs(rp["OR"] - s["OR"]) < 5e-4, (level, rp["OR"], s["OR"])
                        row["se_pat"] = se_from_ci(rp["OR_lo"], rp["OR_hi"])
                        row["pat_lo"], row["pat_hi"] = rp["OR_lo"], rp["OR_hi"]
                        row["p_pat"] = p_two(math.log(s["OR"]) / row["se_pat"])
                rows.append(row)
    assert len(rows) == 25, len(rows)

    p_main = [r["p"] for r in rows]
    p_c1 = [r.get("p_boot", r["p"]) for r in rows]
    p_c2 = [r.get("p_pat", r["p"]) for r in rows]
    for r, a, b, c1, c2 in zip(rows, holm(p_main), bonferroni(p_main), holm(p_c1), holm(p_c2)):
        r["p_holm"], r["p_bonf"], r["p_holm_check1"], r["p_holm_check2"] = a, b, c1, c2

    reading = {}
    for db in "me":
        r = next(x for x in rows if x["db"] == db and x["factor"] == "rass" and x["level"] == "light")
        ps = [r["p_holm"], r["p_holm_check1"]] + ([r["p_holm_check2"]] if db == "e" else [])
        reading[db] = classify(ps)
    if reading["m"] == "remains" and reading["e"] == "remains":
        case = "both remain"
    elif reading["m"] == "remains" and reading["e"] == "not clear":
        case = "MIMIC-IV only"
    elif reading["m"] == "not clear" and reading["e"] == "not clear":
        case = "neither"
    else:
        case = "depends: m=%s, e=%s" % (reading["m"], reading["e"])
    opposite = [f'{x["db"]}:{x["factor"]}:{x["level"]}' for x in rows if x["OR"] > 1 and x["p_holm"] < 0.05]

    out = {"addendum": "SAP_addendum_R17.md v1.0", "family_size": len(rows),
           "method": "two-sided Wald test of log OR; SE from the 95% percentile interval; Holm (Bonferroni for reference)",
           "rows": rows, "reading_light": reading, "text_case": case,
           "opposite_direction_after_holm": opposite}
    with open(os.path.join(RES, "r17_mult.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for x in rows:
        extra = ""
        if "p_boot" in x:
            extra = f'  holm(check1) {x["p_holm_check1"]:.4f}'
        if "p_pat" in x:
            extra += f'  holm(check2) {x["p_holm_check2"]:.4f}'
        print(f'{x["db"]} {x["factor"]:5s} {x["level"]:8s} OR {x["OR"]:.2f} ({x["lo"]:.2f}-{x["hi"]:.2f}) '
              f'p {x["p"]:.2e}  holm {x["p_holm"]:.4f}  bonf {x["p_bonf"]:.4f}{extra}')
    print("reading (awake or lightly sedated):", reading, "| text case:", case, "| opposite:", opposite)


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
