from pathlib import Path
from itertools import product
import sys

from datasets import load_from_disk


def find_repo_root(start: Path) -> Path:
    """
    Find the ChRIMP repo root from the current file location.
    Works even if this script is inside scripts/test_stereo_preprocess/.
    """
    for path in [start, *start.parents]:
        if (path / "scripts" / "annotate_hf_mech_uspto_31k_th.py").exists():
            return path

    raise RuntimeError("Could not find ChRIMP repository root.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())

# Make src/ and scripts/ importable
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annotate_hf_mech_uspto_31k_th import (  # noqa: E402
    collect_stereo_events,
    with_stereo_updates,
    validate_model_output,
)
from chrimp.world.mechsmiles import MechSmiles  # noqa: E402


DATASET_PATH = REPO_ROOT / "data" / "th_annotation_test_500"


def get_predicted_products(candidate: str) -> tuple[str, str]:
    """
    Return both mapped and unmapped predicted products when possible.
    """
    msmi = MechSmiles(candidate)

    try:
        mapped_product = msmi.ms_prod.mapped_smiles
    except Exception as exc:
        mapped_product = f"ERROR: {type(exc).__name__}: {exc}"

    try:
        unmapped_product = msmi.prod
    except Exception as exc:
        unmapped_product = f"ERROR: {type(exc).__name__}: {exc}"

    return mapped_product, unmapped_product


def print_predictions_for_row(row):
    mech_smi = row["mech_smi_min"]
    target_smiles = row["target_smiles_for_validation"]

    print("\n==============================")
    print("rxn_idx:", row.get("rxn_idx"))
    print("step_idx_forward:", row.get("step_idx_forward"))
    print("th_status:", row.get("th_status"))
    print("th_event_types:", row.get("th_event_types"))
    print("th_error:", row.get("th_error"))

    print("\nOriginal mech_smi_min:")
    print(mech_smi)

    print("\nTarget smiles:")
    print(target_smiles)

    msmi = MechSmiles(mech_smi)
    events = collect_stereo_events(msmi)

    if not events:
        print("\nNo stereo events detected.")
        return

    print("\nDetected events:")
    for event in events:
        print(event)

    keep_stereo_center_maps = {
        event.center_map
        for event in events
        if event.event_type == "tetrahedral_acceptor"
    }

    compare_stereo = bool(keep_stereo_center_maps)

    print("\nStereo centers kept during validation:")
    print(keep_stereo_center_maps)

    print("\ncompare_stereo:")
    print(compare_stereo)

    print("\nCandidate predictions:")

    for modes in product(*(event.mode_options for event in events)):
        candidate = with_stereo_updates(mech_smi, events, modes)

        mapped_product, unmapped_product = get_predicted_products(candidate)

        matches_target, error = validate_model_output(
            candidate,
            target_smiles,
            keep_stereo_center_maps=keep_stereo_center_maps,
            compare_stereo=compare_stereo,
        )

        print("\n--- candidate ---")
        print("modes:", modes)

        print("\ncandidate mechsmiles:")
        print(candidate)

        print("\npredicted mapped product:")
        print(mapped_product)

        print("\npredicted unmapped product:")
        print(unmapped_product)

        print("\nmatches target:")
        print(matches_target)

        print("\nvalidation error:")
        print(error)


def find_row(ds, rxn_idx, step_idx_forward=None):
    for split_name, split in ds.items():
        for row in split:
            if row["rxn_idx"] != rxn_idx:
                continue

            if (
                step_idx_forward is not None
                and row["step_idx_forward"] != step_idx_forward
            ):
                continue

            print(f"\nFound in split: {split_name}")
            return row

    return None


if __name__ == "__main__":
    ds = load_from_disk(str(DATASET_PATH))

    # Change these values to inspect another reaction
    RXN_IDX = 10065
    STEP_IDX_FORWARD = 3

    row = find_row(ds, RXN_IDX, STEP_IDX_FORWARD)

    if row is None:
        print("No matching row found.")
    else:
        print_predictions_for_row(row)