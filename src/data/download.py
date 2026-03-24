from pathlib import Path
import requests


def download_rcsb_cif(pdb_id, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{pdb_id.lower()}.cif"
    if out_path.exists():
        return str(out_path), "exists"

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    resp = requests.get(url, timeout=60)

    if resp.status_code != 200:
        return None, f"failed_http_{resp.status_code}"

    with open(out_path, "wb") as f:
        f.write(resp.content)

    return str(out_path), "downloaded"