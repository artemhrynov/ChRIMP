from pathlib import Path
from itertools import product
import sys

from datasets import load_from_disk

# Make scripts importable
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annotate_hf_mech_uspto_31k_th import (
    collect_stereo_events,
    with_stereo_updates,
    validate_model_output,
)
from chrimp.world.mechsmiles import MechSmiles


DATASET_PATH = "data/th_annotation_test_500"  # change if needed


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

    ignore_stereo_center_maps = {
        event.center_map
        for event in events
        if event.event_type == "planar_to_tetrahedral"
    }

    print("\nIgnored stereo center maps during validation:")
    print(ignore_stereo_center_maps)

    print("\nCandidate predictions:")

    for modes in product(*(event.mode_options for event in events)):
        candidate = with_stereo_updates(mech_smi, events, modes)

        try:
            predicted_product = MechSmiles(candidate).prod
        except Exception as exc:
            predicted_product = f"ERROR: {type(exc).__name__}: {exc}"

        matches_target, error = validate_model_output(
            candidate,
            target_smiles,
            ignore_stereo_center_maps=ignore_stereo_center_maps,
        )

        print("\n--- candidate ---")
        print("modes:", modes)
        print("candidate mechsmiles:")
        print(candidate)
        print("predicted product:")
        print(predicted_product)
        print("matches target:", matches_target)
        print("validation error:", error)


def find_row(ds, rxn_idx, step_idx_forward=None):
    for split_name, split in ds.items():
        for row in split:
            if row["rxn_idx"] != rxn_idx:
                continue

            if step_idx_forward is not None and row["step_idx_forward"] != step_idx_forward:
                continue

            print(f"\nFound in split: {split_name}")
            return row

    return None


if __name__ == "__main__":
    ds = load_from_disk(DATASET_PATH)

    # Change these values to the reaction you want to inspect
    RXN_IDX = 10680
    STEP_IDX_FORWARD = 0  # or put a number, for example 0, 1, 2...

    row = find_row(ds, RXN_IDX, STEP_IDX_FORWARD)

    if row is None:
        print("No matching row found.")
    else:
        print_predictions_for_row(row)