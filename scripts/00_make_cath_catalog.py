from pathlib import Path
import csv

DOMAIN_LIST = Path("data/raw/cath/cath-domain-list.txt")
BOUNDARIES = Path("data/raw/cath/cath-domain-boundaries.txt")
S40_LIST = Path("data/raw/cath/cath-dataset-nonredundant-S40.list")
OUT_CSV = Path("data/raw/catalog/all_beta_catalog.csv")


def load_s40_ids(path: Path) -> set[str]:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line.split()[0])
    return ids


def load_mainly_beta_domain_ids(domain_list_path: Path, allowed_ids: set[str] | None = None) -> set[str]:
    """
    cath-domain-list.txt format:
    col1 domain_id
    col2 class
    ...
    We keep only class == 2 (Mainly Beta).
    """
    beta_ids = set()

    with open(domain_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            domain_id = parts[0]
            cath_class = parts[1]

            if cath_class != "2":
                continue

            if allowed_ids is not None and domain_id not in allowed_ids:
                continue

            beta_ids.add(domain_id)

    return beta_ids


def parse_boundaries_line(line: str):
    """
    CDF format (simplified for our use):
    chain_name Dxx Fxx [domain blocks...] [fragment blocks...]

    Each domain block starts with:
      N  C  S  I  C  E  I   (for 1 segment)
    or
      N  [segment1] [segment2] ...  (for multiple segments)

    We only keep single-segment domains because current pipeline expects:
      domain_id,pdb_id,chain_id,start_res,end_res
    """
    parts = line.split()
    chain_name = parts[0]          # e.g. 1abcA
    n_domains = int(parts[1][1:])  # D02 -> 2
    n_frags = int(parts[2][1:])    # F00 -> 0

    idx = 3
    parsed_domains = []

    for domain_num in range(1, n_domains + 1):
        n_segments = int(parts[idx])
        idx += 1

        segments = []
        for _ in range(n_segments):
            # segment tokens: C S I C E I
            start_chain = parts[idx]
            start_res = parts[idx + 1]
            start_ins = parts[idx + 2]
            end_chain = parts[idx + 3]
            end_res = parts[idx + 4]
            end_ins = parts[idx + 5]
            idx += 6

            segments.append(
                {
                    "start_chain": start_chain,
                    "start_res": start_res,
                    "start_ins": start_ins,
                    "end_chain": end_chain,
                    "end_res": end_res,
                    "end_ins": end_ins,
                }
            )

        parsed_domains.append(
            {
                "domain_index": domain_num,
                "n_segments": n_segments,
                "segments": segments,
            }
        )

    # Skip fragment tokens, we do not need them
    # Each fragment has 7 tokens: C S I C E I NR
    idx += n_frags * 7

    return chain_name, parsed_domains


def domain_id_from_chain_and_index(chain_name: str, domain_index: int) -> str:
    """
    chain_name: 5 chars, e.g. 1abcA
    domain IDs are 7 chars, e.g. 1abcA01
    """
    return f"{chain_name}{domain_index:02d}"


def build_catalog(boundaries_path: Path, beta_ids: set[str]):
    rows = []

    with open(boundaries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                chain_name, domains = parse_boundaries_line(line)
            except Exception:
                continue

            pdb_id = chain_name[:4].lower()
            chain_id = chain_name[4]

            # Current DSSP pipeline is safer with explicit chain IDs
            if chain_id == "0":
                continue

            for d in domains:
                domain_id = domain_id_from_chain_and_index(chain_name, d["domain_index"])

                if domain_id not in beta_ids:
                    continue

                # Keep only contiguous single-segment domains
                if d["n_segments"] != 1:
                    continue

                seg = d["segments"][0]

                try:
                    start_res = int(seg["start_res"])
                    end_res = int(seg["end_res"])
                except ValueError:
                    continue

                if end_res < start_res:
                    continue

                rows.append(
                    {
                        "domain_id": domain_id,
                        "pdb_id": pdb_id,
                        "chain_id": chain_id,
                        "start_res": start_res,
                        "end_res": end_res,
                    }
                )

    # remove duplicates
    unique = {}
    for r in rows:
        unique[r["domain_id"]] = r

    rows = list(unique.values())
    rows.sort(key=lambda x: x["domain_id"])
    return rows


def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain_id", "pdb_id", "chain_id", "start_res", "end_res"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not DOMAIN_LIST.exists():
        raise FileNotFoundError(DOMAIN_LIST)
    if not BOUNDARIES.exists():
        raise FileNotFoundError(BOUNDARIES)
    if not S40_LIST.exists():
        raise FileNotFoundError(S40_LIST)

    s40_ids = load_s40_ids(S40_LIST)
    beta_ids = load_mainly_beta_domain_ids(DOMAIN_LIST, allowed_ids=s40_ids)
    rows = build_catalog(BOUNDARIES, beta_ids)
    write_csv(rows, OUT_CSV)

    print(f"S40 total ids: {len(s40_ids)}")
    print(f"Mainly-beta S40 ids: {len(beta_ids)}")
    print(f"Single-segment catalog rows written: {len(rows)}")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()