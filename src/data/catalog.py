import pandas as pd

REQUIRED_COLS = ["domain_id", "pdb_id", "chain_id", "start_res", "end_res"]

COLUMN_ALIASES = {
    "id": "domain_id",
    "domain": "domain_id",
    "pdb": "pdb_id",
    "chain": "chain_id",
    "start": "start_res",
    "end": "end_res"
}


def normalize_columns(df):
    new_cols = {}
    for c in df.columns:
        c_clean = c.strip().lower()
        new_cols[c] = COLUMN_ALIASES.get(c_clean, c_clean)
    return df.rename(columns=new_cols)


def load_catalog(path):
    df = pd.read_csv(path)
    df = normalize_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar var: {missing}")

    df["domain_id"] = df["domain_id"].astype(str)
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower().str.strip()
    df["chain_id"] = df["chain_id"].astype(str).str.strip()
    df["start_res"] = df["start_res"].astype(int)
    df["end_res"] = df["end_res"].astype(int)

    df = df.drop_duplicates(subset=["domain_id"]).reset_index(drop=True)
    return df


def build_manifests(df, train_candidate_pool, test_candidate_pool, random_seed):
    if len(df) < train_candidate_pool + test_candidate_pool:
        raise ValueError(
            f"Katalogda yeterli örnek yok. Gerekli en az sayı: "
            f"{train_candidate_pool + test_candidate_pool}, mevcut: {len(df)}"
        )

    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    train_df = shuffled.iloc[:train_candidate_pool].copy()
    remaining = shuffled.iloc[train_candidate_pool:].copy()
    test_df = remaining.iloc[:test_candidate_pool].copy()

    return train_df, test_df