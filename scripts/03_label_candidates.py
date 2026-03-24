import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.utils.io_utils import load_yaml, write_jsonl
from src.data.dssp_labeler import extract_domain_sequence_and_labels

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence_and_labels(sequence, labels, residue_numbers):
    new_seq = []
    new_labels = []
    new_resnums = []

    for aa, lab, rn in zip(sequence, labels, residue_numbers):
        if aa in AA20 and lab in {"B", "N"}:
            new_seq.append(aa)
            new_labels.append(lab)
            new_resnums.append(rn)

    return "".join(new_seq), new_labels, new_resnums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV manifest yolu")
    parser.add_argument("--output", required=True, help="JSONL çıktı yolu")
    args = parser.parse_args()

    cfg = load_yaml("configs/config.yaml")
    df = pd.read_csv(args.manifest)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Labeling {Path(args.manifest).name}"):
        pdb_id = str(row["pdb_id"]).lower().strip()
        chain_id = str(row["chain_id"]).strip()
        start_res = int(row["start_res"])
        end_res = int(row["end_res"])
        domain_id = str(row["domain_id"])

        cif_path = Path(cfg["structures_dir"]) / f"{pdb_id}.cif"
        if not cif_path.exists():
            continue

        try:
            out = extract_domain_sequence_and_labels(
                cif_path=str(cif_path),
                pdb_id=pdb_id,
                chain_id=chain_id,
                start_res=start_res,
                end_res=end_res,
                dssp_exec=cfg["dssp_exec"]
            )

            seq, labels, resnums = clean_sequence_and_labels(
                out["sequence"],
                out["labels"],
                out["residue_numbers"]
            )

            if len(seq) == 0:
                continue

            if len(seq) != len(labels):
                continue

            rec = {
                "domain_id": domain_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "start_res": start_res,
                "end_res": end_res,
                "sequence": seq,
                "labels": labels,
                "residue_numbers": resnums,
                "length": len(seq),
                "beta_count": sum(1 for x in labels if x == "B"),
                "nonbeta_count": sum(1 for x in labels if x == "N")
            }
            records.append(rec)

        except Exception as e:
            print(f"[FAIL] {domain_id} -> {e}")

    write_jsonl(records, args.output)
    print("Yazıldı:", args.output)
    print("Başarılı örnek sayısı:", len(records))


if __name__ == "__main__":
    main()