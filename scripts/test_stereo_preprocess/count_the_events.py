from datasets import load_from_disk
from collections import Counter

DATASET_PATH = "data/th_annotation_test_500"  # change if needed

ds = load_from_disk(DATASET_PATH)

for split_name, split in ds.items():
    print(f"\n===== {split_name} =====")

    counts = Counter(
        (row["th_status"], row["th_event_types"])
        for row in split
    )

    for key, count in counts.most_common():
        print(key, count)