import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asd

EXPECT = {
    ("mimic", "c1_cohort"): "subject_id hadm_id stay_id intime outtime gender era site age t0 scr egfr daily arm",
    ("mimic", "c2_cohort"): "subject_id hadm_id stay_id intime outtime gender era site age t0 scr egfr daily arm",
    ("eicu", "c1_cohort"): "subject_id stay_id gender era site age t0 scr egfr daily arm",
    ("eicu", "c2_cohort"): "subject_id stay_id gender era site age t0 scr egfr daily arm",
    ("mimic", "covars"): "stay_id sofa rass vent vaso sed_inf opi_inf inr_max plt_min unit service route h_icu dementia alcohol liver",
    ("eicu", "covars"): "stay_id apache vent rass sed_inf opi_inf vaso inr_max plt_min unit service route h_icu dementia alcohol liver",
    ("mimic", "follow"): "stay_id t_end end_death", ("eicu", "follow"): "stay_id t_end end_death",
    ("mimic", "c1_dev"): "stay_id t_dev", ("mimic", "c2_dev"): "stay_id t_dev",
    ("eicu", "c1_dev"): "stay_id t_dev", ("eicu", "c2_dev"): "stay_id t_dev",
    ("mimic", "outcome"): "stay_id t_del t_hypo hosp_death", ("eicu", "outcome"): "stay_id t_del t_hypo hosp_death",
}
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)


def empty_tables(con, src, mods):
    n = 0
    for m in mods:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {m}")
        for f in sorted(os.listdir(os.path.join(src, m))):
            path = os.path.join(src, m, f)
            if f.endswith(".parquet"):
                name, ref = f[:-8], path
            elif os.path.isdir(path) and any(x.endswith(".parquet") for x in os.listdir(path)):
                name, ref = f, os.path.join(path, "*.parquet")
            else:
                continue
            cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{ref}')").fetchall()
            con.execute(f'CREATE TABLE {m}."{name}" (' + ", ".join(f'"{c[0]}" {c[1]}' for c in cols) + ")")
            n += 1
    return n


def run_db(db, src, tmp):
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads = 2")
    mods = ("hosp", "icu", "ed", "note", "derived") if db == "mimic" else ("eicu", "eicu_derived")
    n = empty_tables(con, src, mods)
    check(f"{db}: empty tables with the real schemas", n > 0, f"{n} tables")
    for step, specs in asd.slim_specs(db, None, lambda m, tb: f'{m}."{tb}"').items():
        for m, tb, where, _ in specs:
            try:
                con.execute(f'SELECT * FROM {m}."{tb}" WHERE {where} LIMIT 0').fetchall()
                check(f"{db}: slim step {step}, {m}.{tb}", True)
            except Exception as e:
                check(f"{db}: slim step {step}, {m}.{tb}", False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
    d = "m" if db == "mimic" else "e"
    for part in ("cohort", "covars", "follow", "outcome"):
        for sql, path in asd.all_sql(tmp, db)[part]:
            name = os.path.basename(path)[2:-8]
            try:
                con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")
                cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
                want = EXPECT[(db, name)].split()
                check(f"{db}: extract {part} -> {d}_{name}", cols == want, "columns as the analysis reads them" if cols == want else f"got {cols}")
            except Exception as e:
                check(f"{db}: extract {part} -> {d}_{name}", False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
    con.close()


def main():
    t0 = time.time()
    if len(sys.argv) != 3:
        raise SystemExit(1)
    tmp = tempfile.mkdtemp(prefix="asd_bind_")
    try:
        for db, src in (("mimic", sys.argv[1]), ("eicu", sys.argv[2])):
            run_db(db, os.path.expanduser(src), tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
