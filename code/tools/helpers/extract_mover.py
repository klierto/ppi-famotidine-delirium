import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import tarfile
import time


def _hook(t, e, tb):
    print(f"ERROR {t.__name__}: {str(e).splitlines()[0] if str(e) else ''}", file=sys.stderr)


sys.excepthook = _hook

EMR = "EPIC_EMR.tar.gz"
SIS = "sis_emr.tar.gz"
FLO = {"cleaned": "Epic_flowsheets_cleaned.tar.gz", "raw": "EPIC_patient_measurments.tar.gz"}
FLO_KEEP = ["LOG_ID", "MRN", "FLO_NAME", "FLO_DISPLAY_NAME", "RECORD_TYPE", "RECORDED_TIME", "MEAS_VALUE", "UNITS"]
STREAMED = {"meds": (EMR, r"patient_medications\.csv$", "patient_medications"),
            "labs": (EMR, r"patient_labs\.csv$", "patient_labs")}


def table_name(member):
    return re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(member))[0].lower()).strip("_")


def norm_header(cols):
    out = []
    for i, c in enumerate(cols):
        c = c.strip() or f"col{i}"
        out.append(c if c not in out else f"{c}_{i}")
    return out


def read_csv_varchar(con, dst, path):
    base = f"read_csv('{path}', header=true, all_varchar=true, quote='\"', escape='\"'"
    try:
        con.execute(f"CREATE OR REPLACE TABLE {dst} AS SELECT * FROM {base}, sample_size=-1)")
    except Exception:
        con.execute(f"CREATE OR REPLACE TABLE {dst} AS SELECT * FROM {base}, delim=',', strict_mode=false, null_padding=true)")


US_TIME = "['%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%m/%d/%y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y', '%m/%d/%y']"
ISO_TIME = r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?$"


def retype(con, src, dst):
    cols = [r[0] for r in con.execute(f"DESCRIBE {src}").fetchall()]
    probes = ", ".join(
        f'count("{c}") - count(CASE WHEN regexp_full_match("{c}", \'-?[0-9]+\') THEN TRY_CAST("{c}" AS BIGINT) END), '
        f'count("{c}") - count(TRY_CAST("{c}" AS DOUBLE)), '
        f'count("{c}") - count(CASE WHEN regexp_matches("{c}", \'{ISO_TIME}\') THEN TRY_CAST("{c}" AS TIMESTAMP) END), '
        f'count("{c}") - count(try_strptime("{c}", {US_TIME}))' for c in cols)
    res = con.execute(f"SELECT {probes} FROM {src}").fetchone()
    exprs = []
    for i, c in enumerate(cols):
        bad_int, bad_dbl, bad_iso, bad_us = res[4 * i:4 * i + 4]
        low = c.lower()
        if low in ("mrn", "log_id", "pid") or low.endswith("_id") or low.endswith("_name") or low.endswith("_nm"):
            exprs.append(f'"{c}"')
        elif bad_int == 0:
            exprs.append(f'TRY_CAST("{c}" AS BIGINT) AS "{c}"')
        elif bad_dbl == 0:
            exprs.append(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"')
        elif bad_iso == 0:
            exprs.append(f'TRY_CAST("{c}" AS TIMESTAMP) AS "{c}"')
        elif bad_us == 0:
            exprs.append(f'try_strptime("{c}", {US_TIME}) AS "{c}"')
        else:
            exprs.append(f'"{c}"')
    con.execute(f"CREATE OR REPLACE TABLE {dst} AS SELECT {', '.join(exprs)} FROM {src}")


def load_members(con, archive, schema, wanted_re, stage):
    os.makedirs(stage, exist_ok=True)
    done = []
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".csv") or not re.search(wanted_re, m.name):
                continue
            tmp = os.path.join(stage, "member.csv")
            with tf.extractfile(m) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            name = table_name(m.name)
            read_csv_varchar(con, f"{schema}._raw_tbl", tmp)
            retype(con, f"{schema}._raw_tbl", f'{schema}."{name}"')
            con.execute(f"DROP TABLE {schema}._raw_tbl")
            os.remove(tmp)
            done.append(name)
            print(f"  {schema}.{name}: {con.execute(f'SELECT count(*) FROM {schema}.{name}').fetchone()[0]} rows", flush=True)
    return done


class TarCsvStream:

    def __init__(self, archive, member_re, state_path, index_path):
        self.archive, self.member_re = archive, member_re
        self.state_path, self.index_path = state_path, index_path
        self.st = {"pos": 0, "member": None, "end": 0, "end_pad": 0, "cols": None, "done": False,
                   "members_done": [], "bytes_done": 0}
        if os.path.exists(state_path):
            self.st = json.load(open(state_path))

    def _header(self, f, pos):
        f.seek(pos)
        pax = {}
        while True:
            buf = f.read(512)
            if len(buf) < 512 or buf == b"\0" * 512:
                return None
            ti = tarfile.TarInfo.frombuf(buf, "utf-8", "surrogateescape")
            data = pos + 512
            pad = (ti.size + 511) // 512 * 512
            if ti.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
                payload = f.read(pad)[:ti.size]
                if ti.type == tarfile.GNUTYPE_LONGNAME:
                    pax["path"] = payload.rstrip(b"\0").decode("utf-8", "replace")
                else:
                    for rec in payload.split(b"\n"):
                        if b"=" in rec:
                            k, v = rec.split(b" ", 1)[1].split(b"=", 1)
                            pax[k.decode()] = v.decode("utf-8", "replace")
                pos = data + pad
                continue
            size = int(pax.get("size", ti.size))
            name = pax.get("path", ti.name)
            return name, ti.isfile(), data, size, data + (size + 511) // 512 * 512

    def run(self, on_chunk, budget_s, chunk=96 << 20):
        import indexed_gzip as igz
        st = self.st
        f = igz.IndexedGzipFile(self.archive, spacing=64 << 20, readbuf_size=16 << 20)
        if os.path.exists(self.index_path):
            f.import_index(self.index_path)
        t0, done_here = time.time(), 0
        carry, read_pos = b"", None
        while not st["done"]:
            if done_here and time.time() - t0 >= budget_s:
                break
            if st["member"] is None:
                h = self._header(f, st["pos"])
                if h is None:
                    st["done"] = True
                    break
                name, isfile, data, size, end_pad = h
                if not isfile or size == 0 or not re.search(self.member_re, name):
                    st["pos"] = end_pad
                    continue
                f.seek(data)
                head = f.read(min(size, 1 << 20))
                nl = head.find(b"\n")
                st["cols"] = norm_header(next(csv.reader([head[:nl].decode("utf-8-sig", "replace").rstrip("\r")])))
                st.update(member=name, end=data + size, end_pad=end_pad, pos=data + nl + 1)
                carry, read_pos = b"", None
            if read_pos is None:
                f.seek(st["pos"])
                read_pos = st["pos"]
            want = min(chunk, st["end"] - read_pos)
            got = f.read(want) if want > 0 else b""
            if want > 0 and not got:
                raise RuntimeError("archive ended inside a member")
            data_bytes = carry + got
            read_pos += len(got)
            if read_pos < st["end"]:
                cut = data_bytes.rfind(b"\n")
                if cut < 0:
                    raise RuntimeError("line longer than the read chunk")
                body, carry = data_bytes[:cut + 1], data_bytes[cut + 1:]
            else:
                body, carry = data_bytes, b""
            if body:
                on_chunk(body, st["cols"], st["member"])
            st["pos"] += len(body)
            st["bytes_done"] += len(body)
            done_here += len(body)
            if read_pos >= st["end"] and not carry:
                st["members_done"].append(st["member"])
                st.update(member=None, pos=st["end_pad"])
                read_pos = None
        f.export_index(self.index_path + ".tmp")
        os.replace(self.index_path + ".tmp", self.index_path)
        f.close()
        json.dump(st, open(self.state_path + ".tmp", "w"))
        os.replace(self.state_path + ".tmp", self.state_path)
        return st


def make_filter(stage_prefix, keep_cols, logs, mrns, flo_regex, stats):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    log_set = pa.array(sorted(logs), type=pa.string())
    mrn_set = pa.array(sorted(mrns), type=pa.string())

    def norm(col):
        return pc.replace_substring_regex(pc.utf8_trim_whitespace(col), r"\.0+$", "")

    def skip(row):
        stats["bad_rows"] += 1
        return "skip"

    def on_chunk(body, cols, member):
        keep = [c for c in keep_cols if c in cols] if keep_cols else cols
        ro = pcsv.ReadOptions(column_names=cols, block_size=8 << 20)
        po = pcsv.ParseOptions(newlines_in_values=False, invalid_row_handler=skip)
        co = pcsv.ConvertOptions(column_types={c: pa.string() for c in cols}, include_columns=keep, strings_can_be_null=True)
        tbl = pcsv.read_csv(pa.py_buffer(body), read_options=ro, parse_options=po, convert_options=co)
        stats["rows_in"] += tbl.num_rows
        mask = None
        if "LOG_ID" in keep:
            mask = pc.is_in(norm(tbl["LOG_ID"]), value_set=log_set)
        mcol = next((c for c in keep if c.lower() == "mrn"), None)
        if mcol and len(mrn_set):
            m2 = pc.is_in(norm(tbl[mcol]), value_set=mrn_set)
            mask = m2 if mask is None else pc.or_kleene(mask, m2)
        if flo_regex:
            names = [c for c in ("FLO_NAME", "FLO_DISPLAY_NAME") if c in keep]
            fm = None
            for c in names:
                x = pc.match_substring_regex(tbl[c], flo_regex, ignore_case=True)
                fm = x if fm is None else pc.or_kleene(fm, x)
            if fm is not None:
                mask = fm if mask is None else pc.and_kleene(mask, fm)
        out = tbl.filter(pc.fill_null(mask, False)) if mask is not None else tbl
        stats["rows_kept"] += out.num_rows
        if out.num_rows:
            for c in keep_cols or []:
                if c not in out.column_names:
                    out = out.append_column(c, pa.nulls(out.num_rows, pa.string()))
            out = out.select(keep_cols or keep)
            for c in [c for c in out.column_names if c.upper() in ("LOG_ID", "MRN")]:
                out = out.set_column(out.column_names.index(c), c, norm(out[c]))
            pq.write_table(out, f"{stage_prefix}_{stats['parts']:06d}.parquet", compression="zstd")
            stats["parts"] += 1

    return on_chunk


def stream_step(con, area, step, archive, member_re, table, keep_cols, flo_regex, budget, stage_dir, idx_dir, chunk, keep_dir):
    st_path = os.path.join(stage_dir, f"{step}.state.json")
    stage_prefix = os.path.join(stage_dir, f"{step}_part")
    idx = os.path.join(idx_dir, os.path.basename(archive) + ".idx")
    logs = {r[0] for r in con.execute("SELECT DISTINCT trim(CAST(LOG_ID AS VARCHAR)) FROM area.cohort WHERE LOG_ID IS NOT NULL").fetchall()}
    mrns = set() if step == "flowsheets" else {r[0] for r in con.execute("SELECT DISTINCT trim(CAST(MRN AS VARCHAR)) FROM area.cohort WHERE MRN IS NOT NULL").fetchall()}
    stats_path = st_path + ".stats"
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {"rows_in": 0, "rows_kept": 0, "bad_rows": 0, "parts": 0}
    s = TarCsvStream(archive, member_re, st_path, idx)
    st = s.run(make_filter(stage_prefix, keep_cols, logs, mrns, flo_regex, stats), budget, chunk)
    json.dump(stats, open(stats_path, "w"))
    if not st["done"]:
        cur = f"; in {os.path.basename(st['member'])}" if st["member"] else ""
        print(f"{step}: not done ({st['bytes_done'] / 1e9:.1f} GB read, {len(st['members_done'])} members finished{cur}); run again", flush=True)
        return False
    parts = sorted(glob.glob(stage_prefix + "_*.parquet"))
    if parts and step == "flowsheets":
        shutil.rmtree(keep_dir, ignore_errors=True)
        os.makedirs(keep_dir)
        for f in parts:
            shutil.move(f, keep_dir)
        con.execute(f"""CREATE OR REPLACE VIEW epic.flowsheets AS
            SELECT LOG_ID, MRN, FLO_NAME, FLO_DISPLAY_NAME, RECORD_TYPE,
                   coalesce(TRY_CAST(RECORDED_TIME AS TIMESTAMP), try_strptime(RECORDED_TIME, {US_TIME})) AS RECORDED_TIME,
                   MEAS_VALUE, TRY_CAST(MEAS_VALUE AS DOUBLE) AS meas_num, UNITS
            FROM read_parquet('{keep_dir}/*.parquet')""")
    elif parts:
        con.execute(f"CREATE OR REPLACE VIEW epic._raw_view AS SELECT * FROM read_parquet('{stage_prefix}_*.parquet')")
        retype(con, "epic._raw_view", f"epic.{table}")
        con.execute("DROP VIEW epic._raw_view")
        for f in parts:
            os.remove(f)
    else:
        con.execute(f"CREATE OR REPLACE TABLE epic.{table} AS SELECT NULL::BIGINT AS LOG_ID WHERE false")
    n = con.execute(f"SELECT count(*) FROM epic.{table}").fetchone()[0]
    print(f"{step}: done; epic.{table} {n} rows (read {stats['rows_in']} rows, unparsable {stats['bad_rows']})", flush=True)
    for p in (st_path, stats_path):
        os.remove(p)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True)
    ap.add_argument("--mover-dir", default=os.path.join(os.environ.get("DATA_DIR", "data"), "MOVER"))
    ap.add_argument("--work", default=os.path.join(os.environ.get("WORK_DIR", "work"), "mover_extract"), help="private folder for the extract, staging and seek indexes")
    ap.add_argument("--icd10", default="", help="comma-separated ICD-10-CM prefixes defining the area (dots ignored)")
    ap.add_argument("--icd9", default="", help="comma-separated ICD-9-CM prefixes; visit and history lists mix ICD-9 and ICD-10")
    ap.add_argument("--proc-regex", default="", help="regex on PRIMARY_PROCEDURE_NM that also defines the area")
    ap.add_argument("--cohort-sql", help="file with a SELECT returning LOG_ID, MRN from epic tables")
    ap.add_argument("--flo-source", choices=sorted(FLO), default="cleaned")
    ap.add_argument("--flo-regex", default="", help="keep only flowsheet rows whose FLO_NAME or FLO_DISPLAY_NAME match")
    ap.add_argument("--budget", type=float, default=140, help="seconds of streaming per call")
    ap.add_argument("--chunk-kb", type=int, default=96 << 10, help="decompressed bytes parsed per chunk (tests use small values)")
    ap.add_argument("steps", nargs="+", choices=["base", "cohort", "meds", "labs", "flowsheets"])
    a = ap.parse_args()

    import duckdb
    root = os.path.expanduser(a.mover_dir)
    work = os.path.expanduser(a.work)
    stage = os.path.join(work, "mover_stage", a.area)
    os.makedirs(stage, exist_ok=True)
    db = os.path.join(work, f"mover_{a.area}.duckdb")
    con = duckdb.connect(db)
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1500MB'")
    tmp = os.path.join(work, "duck_tmp")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    for s in ("epic", "sis", "area"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
    t_all = time.time()
    for step in a.steps:
        t = time.time()
        if step == "base":
            load_members(con, os.path.join(root, EMR), "epic", r"(?<!medications)(?<!labs)\.csv$", stage)
            load_members(con, os.path.join(root, SIS), "sis", r"\.csv$", stage)
        elif step == "cohort":
            if a.cohort_sql:
                sql = open(a.cohort_sql, encoding="utf-8").read().strip().rstrip(";")
            else:
                codes = [re.sub(r"[^A-Za-z0-9]", "", c).upper() for c in a.icd10.split(",") if c.strip()]
                codes9 = [re.sub(r"[^0-9VvEe]", "", c).upper() for c in a.icd9.split(",") if c.strip()]
                if not codes and not codes9 and not a.proc_regex:
                    raise ValueError("give --icd10, --icd9, --proc-regex or --cohort-sql")
                like10 = lambda col: " OR ".join(f"upper(replace({col}, '.', '')) LIKE '{c}%'" for c in codes)
                like9 = lambda col: " OR ".join(f"(regexp_matches({col}, '^[0-9VE]') AND upper(replace({col}, '.', '')) LIKE '{c}%')"
                                                for c in codes9)
                like = lambda col: " OR ".join(x for x in (like10(col), like9(col)) if x) or "FALSE"
                proc = f"regexp_matches(PRIMARY_PROCEDURE_NM, '{a.proc_regex}', 'i')" if a.proc_regex else "FALSE"
                sql = f"""WITH m AS (
                    SELECT trim(CAST(mrn AS VARCHAR)) AS mrn FROM epic.patient_visit WHERE {like('CAST(diagnosis_code AS VARCHAR)')}
                    UNION SELECT trim(CAST(mrn AS VARCHAR)) FROM epic.patient_history WHERE {like('CAST(diagnosis_code AS VARCHAR)')}
                    UNION SELECT trim(CAST(MRN AS VARCHAR)) FROM epic.patient_coding
                      WHERE REF_BILL_CODE_SET_NAME ILIKE 'ICD-10-CM%' AND ({like10('CAST(REF_BILL_CODE AS VARCHAR)') or 'FALSE'})
                    UNION SELECT trim(CAST(MRN AS VARCHAR)) FROM epic.patient_information WHERE {proc})
                  SELECT DISTINCT p.LOG_ID, trim(CAST(p.MRN AS VARCHAR)) AS MRN
                  FROM epic.patient_information p JOIN m ON m.mrn = trim(CAST(p.MRN AS VARCHAR))"""
            con.execute(f"CREATE OR REPLACE TABLE area.cohort AS {sql}")
            n_log, n_mrn = con.execute("SELECT count(DISTINCT LOG_ID), count(DISTINCT MRN) FROM area.cohort").fetchone()
            print(f"  area.cohort: {n_log} surgical cases of {n_mrn} patients")
        elif step in STREAMED:
            archive, member_re, table = STREAMED[step]
            stream_step(con, a.area, step, os.path.join(root, archive), member_re, table, None, "",
                        a.budget - (time.time() - t_all), stage, os.path.join(work, "mover_stage"), a.chunk_kb << 10, None)
        elif step == "flowsheets":
            stream_step(con, a.area, step, os.path.join(root, FLO[a.flo_source]), r"\.csv$", "flowsheets",
                        FLO_KEEP, a.flo_regex, a.budget - (time.time() - t_all), stage, os.path.join(work, "mover_stage"),
                        a.chunk_kb << 10, os.path.join(work, f"mover_{a.area}_parquet", "flowsheets"))
        print(f"{step}: {time.time() - t:.0f}s", flush=True)
    con.close()


if __name__ == "__main__":
    main()
