from src.utils.io_utils import load_yaml, ensure_dir
from src.data.catalog import load_catalog, build_manifests


def main():
    cfg = load_yaml("configs/config.yaml")

    df = load_catalog(cfg["catalog_csv"])
    train_df, test_df = build_manifests(
        df=df,
        train_candidate_pool=cfg["train_candidate_pool"],
        test_candidate_pool=cfg["test_candidate_pool"],
        random_seed=cfg["random_seed"]
    )

    ensure_dir("data/interim/manifests")
    train_df.to_csv(cfg["train_manifest_csv"], index=False)
    test_df.to_csv(cfg["test_manifest_csv"], index=False)

    print("Train aday manifest yazıldı:", cfg["train_manifest_csv"], len(train_df))
    print("Test aday manifest yazıldı :", cfg["test_manifest_csv"], len(test_df))


if __name__ == "__main__":
    main()