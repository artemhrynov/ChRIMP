from datasets import load_from_disk
import pandas as pd

DATASET_PATH = "data/th_annotation_test_500"

ds = load_from_disk(DATASET_PATH)

rows = []

for split_name, split in ds.items():
    for row in split:
        if row["th_event_types"]:
            rows.append({
                "split": split_name,
                "rxn_idx": row.get("rxn_idx"),
                "step_idx_forward": row.get("step_idx_forward"),
                "th_status": row.get("th_status"),
                "th_event_types": row.get("th_event_types"),
                "th_error": row.get("th_error"),
                "mech_smi_min": row.get("mech_smi_min"),
                "mech_smi_min_th": row.get("mech_smi_min_th"),
                "target_smiles_for_validation": row.get("target_smiles_for_validation"),
            })

df = pd.DataFrame(rows)
df.to_csv("all_stereo_event_examples.csv", index=False)

print(f"Saved {len(df)} stereo-relevant rows.")
print(df[["th_status", "th_event_types"]].value_counts())