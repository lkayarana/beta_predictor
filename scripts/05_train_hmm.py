from pathlib import Path

from src.utils.io_utils import load_yaml, read_jsonl
from src.model.hmm_model import BetaHMM


def main():
    cfg = load_yaml("configs/config.yaml")
    train_records = read_jsonl(cfg["train_jsonl"])

    model = BetaHMM()
    model.fit_supervised(train_records, pseudocount=cfg["pseudocount"])

    out_path = Path(cfg["model_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    print("Model kaydedildi:", out_path)
    print("Start probabilities:")
    print(model.start_probs)
    print("Transition matrix:")
    print(model.trans_probs)


if __name__ == "__main__":
    main()