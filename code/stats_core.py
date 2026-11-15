import math

import numpy as np

H = 168
RIDGE = 1e-3
TRUNC_PCT = 99
SEG = [0, 6, 12, 24, 48, 72, 120]


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def binom_fit(D, succ, trials, ridge=RIDGE, beta0=None, maxit=60, tol=1e-8):
    D = np.asarray(D, float)
    succ = np.asarray(succ, float)
    trials = np.asarray(trials, float)
    n, p = D.shape
    if beta0 is not None and len(beta0) == p:
        beta = np.array(beta0, float)
    else:
        beta = np.zeros(p)
        tot, s = trials.sum(), succ.sum()
        if tot > 0 and 0 < s < tot:
            beta[0] = math.log(s / (tot - s))
        elif tot > 0 and s == 0:
            beta[0] = -12.0
        elif tot > 0:
            beta[0] = 12.0
    pen = np.full(p, ridge)
    pen[0] = 0.0
    for _ in range(maxit):
        mu = expit(D @ beta)
        w = trials * mu * (1.0 - mu)
        g = D.T @ (succ - trials * mu) - pen * beta
        Hm = (D * w[:, None]).T @ D + np.diag(pen) + 1e-9 * np.eye(p)
        try:
            step = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Hm, g, rcond=None)[0]
        mx = float(np.max(np.abs(step))) if p else 0.0
        if mx > 5.0:
            step *= 5.0 / mx
        beta = beta + step
        if mx < tol:
            break
    return beta


def logit_fit(X, y, ridge=RIDGE, beta0=None):
    X = np.asarray(X, float)
    D = np.column_stack([np.ones(len(X)), X])
    return binom_fit(D, np.asarray(y, float), np.ones(len(X)), ridge=ridge, beta0=beta0)


def logit_pred(beta, X):
    return expit(beta[0] + np.asarray(X, float) @ beta[1:])


def hour_index(t):
    t = np.asarray(t, float)
    out = np.full(t.shape, -1, dtype=int)
    ok = np.isfinite(t)
    out[ok] = np.maximum(np.ceil(t[ok]) - 1, 0).astype(int)
    return out


def ps_fit(X, A, beta0=None):
    beta = logit_fit(X, A, beta0=beta0)
    ps = np.clip(logit_pred(beta, X), 1e-9, 1 - 1e-9)
    return ps, beta


def overlap_w(ps, A):
    return np.where(np.asarray(A) == 1, 1.0 - ps, ps)


def iptw_w(ps, A, trunc=(1, 99)):
    A = np.asarray(A)
    pa = float(A.mean())
    w = np.where(A == 1, pa / ps, (1 - pa) / (1 - ps))
    lo, hi = np.percentile(w, trunc)
    return np.clip(w, lo, hi), (float(lo), float(hi)), float(w.max())


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.size and (w ** 2).sum() > 0 else float("nan")


def wmean(x, w):
    sw = w.sum()
    return float((w * x).sum() / sw) if sw > 0 else float("nan")


def smd_table(X, names, A, w=None):
    A = np.asarray(A)
    inA, inB = A == 1, A == 0
    out = {}
    for j, nm in enumerate(names):
        xa, xb = X[inA, j], X[inB, j]
        if xa.size == 0 or xb.size == 0:
            out[nm] = float("nan")
            continue
        sd = math.sqrt((xa.var() + xb.var()) / 2.0)
        if w is None:
            d = xa.mean() - xb.mean()
        else:
            d = wmean(xa, w[inA]) - wmean(xb, w[inB])
        out[nm] = float(d / sd) if sd > 0 else 0.0
    return out


def _segments(bounds, Hh):
    lo = np.asarray(bounds, int)
    hi = np.append(lo[1:], Hh)
    return lo, hi


def hazard_rows(X, kend, ev, bounds=SEG, Hh=H):
    X = np.asarray(X, float)
    n, p = X.shape
    kend = np.minimum(np.asarray(kend, int), Hh - 1)
    ev = np.asarray(ev, int)
    lo, hi = _segments(bounds, Hh)
    S = len(lo)
    ii, ss, tr, sc = [], [], [], []
    for s in range(S):
        trials = np.minimum(kend + 1, hi[s]) - lo[s]
        keep = trials > 0
        if not keep.any():
            continue
        idx = np.nonzero(keep)[0]
        ii.append(idx)
        ss.append(np.full(idx.size, s))
        tr.append(trials[idx])
        sc.append(((ev[idx] == 1) & (kend[idx] >= lo[s]) & (kend[idx] < hi[s])).astype(float))
    if not ii:
        return None
    ii = np.concatenate(ii)
    ss = np.concatenate(ss)
    segd = np.zeros((ii.size, S - 1))
    m = ss > 0
    segd[np.nonzero(m)[0], ss[m] - 1] = 1.0
    D = np.column_stack([np.ones(ii.size), X[ii], segd])
    return D, np.concatenate(sc), np.concatenate(tr).astype(float)


def fit_hazard(X, kend, ev, bounds=SEG, Hh=H, beta0=None):
    X = np.asarray(X, float)
    rows = hazard_rows(X, kend, ev, bounds, Hh)
    ncoef = 1 + X.shape[1] + len(bounds) - 1
    if rows is None:
        beta = np.zeros(ncoef)
        beta[0] = -12.0
        return beta
    D, sc, tr = rows
    return binom_fit(D, sc, tr, beta0=beta0)


def hazard_matrix(beta, X, bounds=SEG, Hh=H):
    X = np.asarray(X, float)
    n, p = X.shape
    S = len(bounds)
    lin = beta[0] + X @ beta[1:1 + p]
    segc = np.concatenate([[0.0], beta[1 + p:1 + p + S - 1]])
    sid = np.clip(np.searchsorted(np.asarray(bounds), np.arange(Hh), side="right") - 1, 0, S - 1)
    return expit(lin[:, None] + segc[sid][None, :])


def inv_cum(P):
    l = -np.log1p(-np.minimum(P, 1.0 - 1e-9))
    W = np.ones_like(P)
    W[:, 1:] = np.exp(np.minimum(np.cumsum(l, axis=1)[:, :-1], 50.0))
    return W


def at_risk_mask(kend, Hh=H):
    return np.arange(Hh)[None, :] <= np.asarray(kend, int)[:, None]


def ipc_weights(X, A, kend, typ, event, by_arm=False, Hh=H, beta0=None):
    X = np.asarray(X, float)
    A = np.asarray(A, int)
    kc = np.where(np.asarray(typ) > 0, np.asarray(kend) - 1, np.asarray(kend))
    ev = np.asarray(event, int)
    n = len(A)
    W = np.ones((n, Hh))
    Z0 = np.zeros((n, 0))
    betas = {}
    b0 = beta0 or {}
    if not by_arm:
        XA = np.column_stack([X, A])
        b = fit_hazard(XA, kc, ev, Hh=Hh, beta0=b0.get("all"))
        betas["all"] = b
        W = inv_cum(hazard_matrix(b, XA, Hh=Hh))
    for a in (1, 0):
        r = A == a
        if not r.any():
            continue
        if by_arm:
            b = fit_hazard(X[r], kc[r], ev[r], Hh=Hh, beta0=b0.get(a))
            betas[a] = b
            W[r] = inv_cum(hazard_matrix(b, X[r], Hh=Hh))
        bn = fit_hazard(Z0[r], kc[r], ev[r], Hh=Hh)
        W[r] = W[r] / inv_cum(hazard_matrix(bn, Z0[r], Hh=Hh))
    W = np.where(at_risk_mask(kend, Hh), W, 1.0)
    return W, betas


def truncate_arm(W, kend, A, pct=TRUNC_PCT, Hh=H):
    W = W.copy()
    R = at_risk_mask(kend, Hh)
    info = {}
    for a in (1, 0):
        rows = np.asarray(A) == a
        v = W[rows][R[rows]]
        if v.size == 0:
            info[a] = (float("nan"), float("nan"))
            continue
        cut = float(np.percentile(v, pct))
        W[rows] = np.minimum(W[rows], cut)
        info[a] = (cut, float(v.max()))
    return W, info


def follow_types(t_end, end_death, t_ev, Hh=H, t_dev=None, disc_competing=False):
    t_end = np.asarray(t_end, float)
    dth = np.asarray(end_death, bool)
    t_ev = np.asarray(t_ev, float)
    n = t_end.size
    end = np.minimum(t_end, float(Hh))
    stop_dev = np.zeros(n, bool)
    if t_dev is not None:
        td = np.asarray(t_dev, float)
        stop_dev = np.isfinite(td) & (td < end)
        end = np.where(stop_dev, td, end)
    evt = np.isfinite(t_ev) & (t_ev <= end) & (t_ev <= Hh)
    typ = np.zeros(n, int)
    ck = np.zeros(n, int)
    typ[evt] = 1
    natural = ~evt & ~stop_dev & (t_end <= Hh)
    typ[natural & dth] = 2
    disc = natural & ~dth & (t_end < Hh)
    if disc_competing:
        typ[disc] = 2
    else:
        ck[disc] = 1
    ck[~evt & stop_dev] = 2
    ck[(typ == 0) & (ck == 0)] = 3
    tt = np.where(evt, t_ev, end)
    kend = np.minimum(hour_index(tt), Hh - 1)
    kend = np.where(kend < 0, 0, kend)
    return kend, typ, ck


def aj_cif(W, kend, typ, Hh=H):
    kend = np.asarray(kend, int)
    typ = np.asarray(typ, int)
    if W.ndim == 1:
        cnt = np.bincount(kend, weights=W, minlength=Hh)[:Hh]
        den = np.cumsum(cnt[::-1])[::-1]
        e1 = typ == 1
        e2 = typ == 2
        num_e = np.bincount(kend[e1], weights=W[e1], minlength=Hh)[:Hh]
        num_c = np.bincount(kend[e2], weights=W[e2], minlength=Hh)[:Hh]
    else:
        den = (W * at_risk_mask(kend, Hh)).sum(0)
        e1 = np.nonzero(typ == 1)[0]
        e2 = np.nonzero(typ == 2)[0]
        num_e = np.bincount(kend[e1], weights=W[e1, kend[e1]], minlength=Hh)[:Hh]
        num_c = np.bincount(kend[e2], weights=W[e2, kend[e2]], minlength=Hh)[:Hh]
    h = np.divide(num_e, den, out=np.zeros(Hh), where=den > 0)
    d = np.divide(num_c, den, out=np.zeros(Hh), where=den > 0)
    S = np.cumprod(1.0 - h - d)
    Sprev = np.concatenate([[1.0], S[:-1]])
    return np.cumsum(Sprev * h)


def effects(rA, rB):
    rd = rA - rB
    rr = rA / rB if rB > 0 else float("nan")
    orr = (rA / (1 - rA)) / (rB / (1 - rB)) if (0 < rA < 1 and 0 < rB < 1) else float("nan")
    return {"riskA": float(rA), "riskB": float(rB), "RD": float(rd), "RR": float(rr), "OR": float(orr)}


def pool_fe(logors, ses):
    b = np.asarray(logors, float)
    s = np.asarray(ses, float)
    ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
    b, s = b[ok], s[ok]
    if b.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    w = 1 / s ** 2
    m = float((w * b).sum() / w.sum())
    se = float(math.sqrt(1 / w.sum()))
    Q = float((w * (b - m) ** 2).sum())
    I2 = float(max(0.0, (Q - (b.size - 1)) / Q)) if Q > 0 else 0.0
    return m, se, Q, I2, int(b.size)


def pool_dl(logors, ses):
    b = np.asarray(logors, float)
    s = np.asarray(ses, float)
    ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
    b, s = b[ok], s[ok]
    if b.size == 0:
        return float("nan"), float("nan"), float("nan")
    w = 1 / s ** 2
    m = (w * b).sum() / w.sum()
    Q = (w * (b - m) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (b.size - 1)) / c) if c > 0 else 0.0
    ws = 1 / (s ** 2 + tau2)
    return float((ws * b).sum() / ws.sum()), float(math.sqrt(1 / ws.sum())), float(tau2)


def ols(D, y):
    return np.linalg.lstsq(D, y, rcond=None)[0]


def cluster_se(D, resid, cl):
    n, k = D.shape
    bread = np.linalg.pinv(D.T @ D)
    u = D * resid[:, None]
    _, inv = np.unique(cl, return_inverse=True)
    G = int(inv.max()) + 1
    Sg = np.zeros((G, k))
    np.add.at(Sg, inv, u)
    meat = Sg.T @ Sg
    adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = adj * bread @ meat @ bread
    return np.sqrt(np.maximum(np.diag(V), 0))


def tsls(Y, A, Z, X, cl):
    Y = np.asarray(Y, float)
    A = np.asarray(A, float)
    Z = np.asarray(Z, float)
    C = np.column_stack([np.ones(len(Y)), np.asarray(X, float)]) if X is not None and np.size(X) else np.ones((len(Y), 1))
    D1 = np.column_stack([C, Z])
    g = ols(D1, A)
    se1 = cluster_se(D1, A - D1 @ g, cl)
    F = float((g[-1] / se1[-1]) ** 2) if se1[-1] > 0 else float("nan")
    D2 = np.column_stack([C, D1 @ g])
    b = ols(D2, Y)
    return {"pi": float(g[-1]), "F": F, "rd": float(b[-1])}


def mask_n(n):
    n = int(n or 0)
    return "<10" if 0 < n < 10 else str(5 * round(n / 5))


def pct_text(num, den, digits=0):
    if den <= 0 or num < 10:
        return "-"
    return f"{100.0 * num / den:.{digits}f}%"
