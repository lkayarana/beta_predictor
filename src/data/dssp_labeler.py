from Bio.PDB import MMCIFParser
from Bio.PDB.DSSP import DSSP

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def ss_to_label(ss_code):
    if ss_code in {"E", "B"}:
        return "B"
    return "N"


def extract_domain_sequence_and_labels(
    cif_path,
    pdb_id,
    chain_id,
    start_res,
    end_res,
    dssp_exec="mkdssp"
):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, cif_path)
    model = structure[0]

    dssp = DSSP(model, cif_path, dssp=dssp_exec)

    seq = []
    labels = []
    residue_numbers = []

    for key in dssp.keys():
        dssp_chain_id, res_id = key
        hetflag, resseq, icode = res_id

        if str(dssp_chain_id).strip() != str(chain_id).strip():
            continue

        try:
            resseq_int = int(resseq)
        except Exception:
            continue

        if resseq_int < start_res or resseq_int > end_res:
            continue

        aa = dssp[key][1]
        ss = dssp[key][2]

        if aa not in AA20:
            continue

        seq.append(aa)
        labels.append(ss_to_label(ss))
        residue_numbers.append(resseq_int)

    return {
        "sequence": "".join(seq),
        "labels": labels,
        "residue_numbers": residue_numbers
    }