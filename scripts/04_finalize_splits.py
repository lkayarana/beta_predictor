import random

from src.utils.io_utils import load_yaml, read_jsonl, write_jsonl


def filter_records(records, min_len, max_len, min_beta_residues):
    cleaned = []
    for rec in records:
        length = len(rec["sequence"])
        beta_count = sum(1 for x in rec["labels"] if x == "B")

        if length < min_len:
            continue
        if length > max_len:
            continue
        if beta_count < min_beta_residues:
            continue

        cleaned.append(rec)
    return cleaned


def main():
    cfg = load_yaml("configs/config.yaml")

    train_candidates = read_jsonl(cfg["labeled_train_jsonl"])
    test_candidates = read_jsonl(cfg["labeled_test_jsonl"])

    train_candidates = filter_records(
        train_candidates,
        cfg["min_len"],
        cfg["max_len"],
        cfg["min_beta_residues"]
    )
    test_candidates = filter_records(
        test_candidates,
        cfg["min_len"],
        cfg["max_len"],
        cfg["min_beta_residues"]
    )

    rng = random.Random(cfg["random_seed"])
    rng.shuffle(train_candidates)
    rng.shuffle(test_candidates)

    if len(train_candidates) < cfg["train_target"]:
        raise ValueError(
            f"Yeterli train örneği yok. Gerekli: {cfg['train_target']}, mevcut: {len(train_candidates)}"
        )

    if len(test_candidates) < cfg["test_target"]:
        raise ValueError(
            f"Yeterli test örneği yok. Gerekli: {cfg['test_target']}, mevcut: {len(test_candidates)}"
        )

    train_final = train_candidates[:cfg["train_target"]]
    test_final = test_candidates[:cfg["test_target"]]

    train_ids = set(x["domain_id"] for x in train_final)
    test_final = [x for x in test_final if x["domain_id"] not in train_ids]

    if len(test_final) < cfg["test_target"]:
        raise ValueError("Train/Test overlap temizlenince test örneği azaldı. Daha büyük aday havuzu seç.")

    test_final = test_final[:cfg["test_target"]]

    write_jsonl(train_final, cfg["train_jsonl"])
    write_jsonl(test_final, cfg["test_jsonl"])

    print("Final train:", len(train_final))
    print("Final test :", len(test_final))


if __name__ == "__main__":
    main()