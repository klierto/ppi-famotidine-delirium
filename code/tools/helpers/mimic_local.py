import glob
import os

SPELLINGS = {
    "hosp": ["hosp", "mimiciv_hosp", ("physionet-data", "mimiciv_3_1_hosp")],
    "icu": ["icu", "mimiciv_icu", ("physionet-data", "mimiciv_3_1_icu")],
    "ed": ["ed", "mimiciv_ed", ("physionet-data", "mimiciv_ed")],
    "note": ["note", "mimiciv_note", ("physionet-data", "mimiciv_note")],
    "ecg": ["ecg", "mimiciv_ecg", ("physionet-data", "mimiciv_ecg")],
    "derived": ["derived", "mimiciv_derived", ("physionet-data", "mimiciv_3_1_derived")],
    "eicu": ["eicu", "eicu_crd", ("physionet-data", "eicu_crd")],
    "eicu_derived": ["eicu_derived", ("physionet-data", "eicu_crd_derived")],
}


def attach_views(con, parquet_dir):
    con.execute("ATTACH IF NOT EXISTS ':memory:' AS \"physionet-data\"")
    made = {}
    for mod, spellings in SPELLINGS.items():
        tables = {os.path.splitext(os.path.basename(f))[0]: f
                  for f in sorted(glob.glob(os.path.join(parquet_dir, mod, "*.parquet")))}
        for d in sorted(glob.glob(os.path.join(parquet_dir, mod, "*", "_done.txt"))):
            tables[os.path.basename(os.path.dirname(d))] = os.path.join(os.path.dirname(d), "*.parquet")
        if not tables:
            continue
        for sp in spellings:
            schema = f'"{sp[0]}".{sp[1]}' if isinstance(sp, tuple) else sp
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            catalog, name = sp if isinstance(sp, tuple) else (None, sp)
            tables_here = {r[0] for r in con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = ? AND database_name = coalesce(?, current_database())",
                [name, catalog]).fetchall()}
            for table, f in tables.items():
                if table not in tables_here:
                    con.execute(f"CREATE OR REPLACE VIEW {schema}.{table} AS SELECT * FROM read_parquet('{f}')")
        made[mod] = len(tables)
    return made


def connect(parquet_dir, memory_limit="2GB", threads=2, temp_dir="~/duck_tmp", database=":memory:"):
    import duckdb
    parquet_dir = os.path.expanduser(parquet_dir)
    if not os.path.isdir(parquet_dir):
        raise SystemExit(f"Parquet folder not found: {parquet_dir}")
    tmp = os.path.expanduser(temp_dir)
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(database)
    con.execute(f"SET memory_limit = '{memory_limit}'; SET threads = {threads}; SET temp_directory = '{tmp}'; "
                "SET preserve_insertion_order = false")
    attach_views(con, parquet_dir)
    return con
