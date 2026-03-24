import pandas as pd
from tqdm import tqdm

from src.utils.io_utils import load_yaml
from src.data.download import download_rcsb_cif


def main():
    cfg = load_yaml("configs/config.yaml")

    train_df = pd.read_csv(cfg["train_manifest_csv"])
    test_df = pd.read_csv(cfg["test_manifest_csv"])

    pdb_ids = sorted(set(
        train_df["pdb_id"].astype(str).str.lower().tolist() +
        test_df["pdb_id"].astype(str).str.lower().tolist()
    ))

    ok = 0
    fail = 0

    for pdb_id in tqdm(pdb_ids, desc="Downloading CIF files"):
        path, status = download_rcsb_cif(pdb_id, cfg["structures_dir"])
        if path is not None:
            ok += 1
        else:
            fail += 1
            print(f"[FAIL] {pdb_id}: {status}")

    print("İndirilen / mevcut yapı sayısı:", ok)
    print("Başarısız:", fail)


if __name__ == "__main__":
    main()