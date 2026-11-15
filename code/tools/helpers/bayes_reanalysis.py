import argparse
import math
import sys

Z95 = 1.959964
Z85 = 1.036433
SKEPTICAL_SD = 0.355
WEAK_SD = 5.0
ROPE = 1.1
FLOOR = 0.5
TARGET = {"benefit": "benefit", "prior": "benefit", "harm": "rules_out_harm"}
TARGET_TEXT = {"benefit": "P(significant benefit)", "rules_out_harm": "P(rules out important harm)"}


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ard_to_or(p0, ard):
    if not 0 < p0 < 1 or not 0 <= ard < p0:
        raise ValueError("needs 0 < baseline risk < 1 and 0 <= ARD < baseline risk")
    p1 = p0 - ard
    return (p1 / (1 - p1)) / (p0 / (1 - p0))


def result_probs(mean, sd, s, mcid):
    c = Z95 * s
    benefit = phi((-c - mean) / sd)
    harm = 1 - phi((c - mean) / sd)
    return {"benefit": benefit, "harm": harm, "inconclusive": 1 - benefit - harm,
            "rules_out_mcid": 1 - phi((math.log(mcid) + c - mean) / sd),
            "rules_out_harm": phi((-math.log(mcid) - c - mean) / sd)}


def errors(theta, s):
    c = Z95 * s
    a_hi, a_lo = (c - theta) / s, (-c - theta) / s
    dens = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    p_hi, p_lo = 1 - phi(a_hi), phi(a_lo)
    mean_abs = theta * p_hi + s * dens(a_hi) - theta * p_lo + s * dens(a_lo)
    return (p_hi if theta < 0 else p_lo) / (p_hi + p_lo), mean_abs / (p_hi + p_lo) / abs(theta)


def fmt_probs(name, r):
    return (f"{name} | P(significant benefit) {r['benefit']:.3f} | P(significant harm) {r['harm']:.3f} | "
            f"P(inconclusive) {r['inconclusive']:.3f} | P(rules out the MCID) {r['rules_out_mcid']:.3f} | "
            f"P(rules out important harm) {r['rules_out_harm']:.3f}")


def floor_check(s, mcid, rct=None, purpose="benefit", floor=FLOOR, require_evidence=True):
    name, mean, sd = priors(mcid, rct)[-1] if rct else priors(mcid)[0]
    key = TARGET[purpose]
    p = result_probs(mean, math.sqrt(sd ** 2 + s ** 2), s, mcid)[key]
    return name, key, p, p >= floor and (bool(rct) or not require_evidence)


def conclusive_prob(mean, sd, s, mcid, purpose="benefit"):
    c, m = Z95 * s, abs(math.log(mcid))
    lo, hi = (m - c, c) if purpose == "harm" else (-c, c - m)
    if lo >= hi:
        return 1.0
    return phi((lo - mean) / sd) + 1 - phi((hi - mean) / sd)


def precision_ratio(s, mcid):
    return Z95 * s / abs(math.log(mcid))


def forecast(s, mcid, rct=None, purpose="benefit"):
    name, mean, sd = priors(mcid, rct)[-1] if rct else priors(mcid)[0]
    total = math.sqrt(sd ** 2 + s ** 2)
    return {"prior": name, "p_target": result_probs(mean, total, s, mcid)[TARGET[purpose]],
            "p_conclusive": conclusive_prob(mean, total, s, mcid, purpose), "precision_ratio": precision_ratio(s, mcid)}


def forecast_line(s, mcid, rct=None, purpose="benefit"):
    f = forecast(s, mcid, rct, purpose)
    return (f"# forecast | primary prior: {f['prior'].replace(' (primary)', '')} | P(target conclusion) {f['p_target']:.3f} | "
            f"P(conclusive: significant either way or a precise null) {f['p_conclusive']:.3f} | "
            f"precision ratio {f['precision_ratio']:.2f} (1.96 se / |ln MCID|; <= 1 is adequate) | MCID {mcid:g}")


def plan(s, mcid, rct=None, purpose="benefit", floor=FLOOR, require_evidence=True):
    out = [f"# what the study can conclude, by true effect (se {s:g})"]
    truths = [("true OR 1.00", 0.0), (f"true OR = MCID {mcid:.2f}", math.log(mcid))]
    if purpose == "harm":
        truths.append((f"true OR = 1/MCID {1 / mcid:.2f}", -math.log(mcid)))
    if rct:
        truths.append((f"true OR = trial estimate {rct[0]:.2f}", math.log(rct[0])))
    for name, theta in truths:
        line = fmt_probs(name, result_probs(theta, s, s, mcid))
        if theta:
            type_s, ratio = errors(theta, s)
            line += f" | type S {type_s:.3f} | exaggeration x{ratio:.2f}"
        out.append(line)
    out.append("# assurance: the same chances averaged over each informative prior")
    for name, mean, sd in priors(mcid, rct):
        if sd != WEAK_SD:
            out.append(fmt_probs(name, result_probs(mean, math.sqrt(sd ** 2 + s ** 2), s, mcid)))
    name, key, p, ok = floor_check(s, mcid, rct, purpose, floor, require_evidence)
    if ok:
        status = "meets floor"
    elif require_evidence and not rct:
        status = "below floor: no evidence-based prior (not invested)"
    else:
        status = "below floor: not invested"
    out.append(f"# floor | {TARGET_TEXT[key]} under the {name} prior | {p:.3f} | floor {floor:.2f} | {status}")
    return out


def se_from_ci(lo, hi):
    return (math.log(hi) - math.log(lo)) / (2 * Z95)


def priors(mcid, rct=None):
    m = math.log(mcid)
    out = [("skeptical (primary)", 0.0, SKEPTICAL_SD), ("weak neutral", 0.0, WEAK_SD),
           ("optimistic", m, abs(m) / Z85), ("pessimistic", -m, abs(m) / Z85)]
    if rct:
        est, lo, hi = rct
        out.append(("evidence-based (half weight)", math.log(est), math.sqrt(2) * se_from_ci(lo, hi)))
    return out


def posterior(y, s, mean, sd):
    w = 1 / s ** 2 + 1 / sd ** 2
    return (y / s ** 2 + mean / sd ** 2) / w, math.sqrt(1 / w)


def summary(mu, sd, mcid, rope=ROPE):
    p = lambda x: phi((math.log(x) - mu) / sd)
    return {"or": math.exp(mu), "lo": math.exp(mu - Z95 * sd), "hi": math.exp(mu + Z95 * sd),
            "p_benefit": p(1), "p_important_benefit": p(mcid), "p_harm": 1 - p(1), "p_important_harm": 1 - p(1 / mcid),
            "p_rope": p(rope) - p(1 / rope)}


def table(y, s, mcid, rct=None, rope=ROPE):
    rows = []
    for name, mean, sd in priors(mcid, rct):
        mu, sdp = posterior(y, s, mean, sd)
        rows.append((name, mean, sd, summary(mu, sdp, mcid, rope)))
    return rows


def fmt(name, mean, sd, r):
    return (f"{name} | prior N({mean:.3f}, {sd:.3f}) | OR {r['or']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) | "
            f"P(OR<1) {r['p_benefit']:.3f} | P(OR<MCID) {r['p_important_benefit']:.3f} | P(OR>1) {r['p_harm']:.3f} | "
            f"P(OR>1/MCID) {r['p_important_harm']:.3f} | P(ROPE) {r['p_rope']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--or", dest="est", type=float, help="estimated OR or HR")
    ap.add_argument("--ci", type=float, nargs=2, metavar=("LO", "HI"), help="its 95%% confidence interval")
    ap.add_argument("--se", type=float, help="standard error of the log estimate (instead of --ci; the gate's se_ row for --plan)")
    ap.add_argument("--mcid", type=float, help="smallest important benefit as an OR or HR")
    ap.add_argument("--rct", type=float, nargs=3, metavar=("OR", "LO", "HI"), help="trial or meta-analysis estimate for the evidence-based prior")
    ap.add_argument("--rope", type=float, default=ROPE, help="range of practical equivalence: 1/ROPE to ROPE (default 1.1)")
    ap.add_argument("--plan", action="store_true",
                    help="planning stage: chance of each conclusion, and posteriors if the estimate lands on no effect or on the MCID")
    ap.add_argument("--purpose", choices=sorted(TARGET), default="benefit",
                    help="study purpose (--plan): the target conclusion is a significant benefit, or ruling out harm for harm questions")
    ap.add_argument("--floor", type=float, default=FLOOR, help=f"investment floor for --plan (default {FLOOR})")
    ap.add_argument("--allow-skeptical", action="store_true",
                    help="earlier rule: let the skeptical prior meet the floor (with --floor 0.20 reproduces those plans)")
    ap.add_argument("--convert", action="store_true", help="print the OR for --ard at --baseline-risk and stop")
    ap.add_argument("--baseline-risk", type=float)
    ap.add_argument("--ard", type=float, help="absolute risk reduction, e.g. 0.03")
    a = ap.parse_args(argv)
    if a.convert:
        if a.baseline_risk is None or a.ard is None:
            ap.error("--convert needs --baseline-risk and --ard")
        print(f"MCID OR {ard_to_or(a.baseline_risk, a.ard):.3f} (baseline risk {a.baseline_risk:g}, ARD {a.ard:g})")
        return 0
    if a.mcid is None or not 0.2 <= a.mcid <= 5 or a.mcid == 1:
        ap.error("--mcid takes an OR or HR between 0.2 and 5 other than 1 (an SMD is not supported: this tool works on ratio scales)")
    if a.rope <= 1:
        ap.error("--rope takes a ratio above 1 (the range is 1/ROPE to ROPE)")
    if not 0 < a.floor < 1:
        ap.error("--floor takes a probability between 0 and 1")
    if a.est is not None and a.est <= 0 or a.ci and not 0 < a.ci[0] < a.ci[1] or a.se is not None and a.se <= 0:
        ap.error("--or must be positive, --ci needs 0 < LO < HI and --se must be positive")
    mcid = min(a.mcid, 1 / a.mcid)
    if a.plan:
        if not a.se:
            ap.error("--plan needs --se")
        for line in plan(a.se, mcid, a.rct, a.purpose, a.floor, not a.allow_skeptical):
            print(line)
        print(forecast_line(a.se, mcid, a.rct, a.purpose))
        for label, est in (("if the estimate is OR 1.00", 1.0), (f"if the estimate is the MCID ({mcid:.2f})", mcid)):
            print(f"# {label}, se {a.se:g}")
            for row in table(math.log(est), a.se, mcid, a.rct, a.rope):
                print(fmt(*row))
        return 0
    if a.est is None or (a.ci is None and a.se is None):
        ap.error("give --or with --ci or --se")
    s = a.se if a.se else se_from_ci(*a.ci)
    print(f"# estimate OR {a.est:g}, se {s:.4f}, MCID {mcid:.2f}, ROPE {1 / a.rope:.2f}-{a.rope:.2f}")
    for row in table(math.log(a.est), s, mcid, a.rct, a.rope):
        print(fmt(*row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
