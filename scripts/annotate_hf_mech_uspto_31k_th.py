"""
Annotate SchwallerGroup/mech_uspto_31k with explicit TH(...) updates.

The source dataset is left untouched. This script loads the Hugging Face
DatasetDict, preserves its splits, and adds mech_smi_min_th plus diagnostic
columns.

Two stereochemical event types are considered:

1. attacks on already tetrahedral chiral acceptors, tested with invert/clear;
2. attacks on prochiral planar carbon centers, such as carbonyl carbons and
   trigonal carbocations, tested with mix/clear when the product center becomes
   tetrahedral stereogenic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from rdkit import Chem

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chrimp.world.mechsmiles import MechSmiles  # noqa: E402

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


DEFAULT_DATASET_NAME = "SchwallerGroup/mech_uspto_31k"
DEFAULT_MODE_OPTIONS = ("invert", "clear") #should be removed later, different logic is used
SUPPORTED_MODE_OPTIONS = {"invert", "clear", "unknown", "mix"}
TETRAHEDRAL_ACCEPTOR_MODES = ("invert", "clear")
PLANAR_TO_TETRAHEDRAL_MODES = ("mix", "clear")

KEEP_COLUMNS = [
    "mech_smi_min",
    "mech_smi_equ",
    "elem_reac_min",
    "elem_prod_min",
    "elem_reac_equ",
    "elem_prod_equ",
    "rxn_prod_min",
    "rxn_idx",
    "step_idx_forward",
    "step_idx_retro",
    "split",
]
ADDED_COLUMNS = [
    "mech_smi_min_th",
    "th_status",
    "th_error",
    "target_smiles_for_validation",
    "th_event_types",
]
REQUIRED_COLUMNS = {"mech_smi_min", "rxn_idx", "step_idx_forward", "rxn_prod_min"}


@dataclass(frozen=True)
class StereoEvent:
    center_map: int
    ligand_pairs: tuple[tuple[int, int], ...]
    event_type: str
    mode_options: tuple[str, ...]
    


@dataclass(frozen=True)
class InferenceResult:
    mech_smi_min_th: str
    th_status: str
    th_error: str = ""
    th_event_types: tuple[str, ...] = ()


def split_mech_smi(mech_smi: str) -> tuple[str, str, str]:
    parts = str(mech_smi).split("|", 2)
    smiles = parts[0]
    arrows = parts[1] if len(parts) > 1 else ""
    stereo = parts[2] if len(parts) > 2 else ""
    return smiles, arrows, stereo


def canonical_unmapped_smiles(
    smiles: str,
    ignore_stereo_center_maps: set[int] | None = None,
) -> str | None:
    if not isinstance(smiles, str) or not smiles:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            mol.UpdatePropertyCache(strict=False)

    ignore_stereo_center_maps = ignore_stereo_center_maps or set()

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in ignore_stereo_center_maps:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def canonical_component_counter(
    smiles: str,
    ignore_stereo_center_maps: set[int] | None = None,
) -> Counter[str] | None:
    canonical = canonical_unmapped_smiles(
        smiles,
        ignore_stereo_center_maps=ignore_stereo_center_maps,
    )
    if canonical is None:
        return None
    if canonical == "":
        return Counter()
    return Counter(canonical.split("."))


def build_step_targets(df: pd.DataFrame) -> dict[int, str]: #understan this function better
    targets: dict[int, str] = {}

    for _, group in df.groupby("rxn_idx", sort=False):
        ordered = group.sort_values("step_idx_forward", kind="mergesort")
        ordered_indices = list(ordered.index)

        for position, row_index in enumerate(ordered_indices):
            if position + 1 < len(ordered_indices):
                next_mech_smi = df.at[ordered_indices[position + 1], "mech_smi_min"]
                targets[row_index] = split_mech_smi(next_mech_smi)[0]
            else:
                target = df.at[row_index, "rxn_prod_min"]
                targets[row_index] = target if isinstance(target, str) else ""

    return targets

#Helper extracts the ligand replacement for tetrahedral event
def find_ligand_pairs(
    center_idx: int,
    new_ligand_idx: int,
    broken_bonds_by_atom: dict[int, list[int]],
    broken_bond_cursors: dict[int, int],
    idx_to_map: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    broken_ligands = broken_bonds_by_atom.get(center_idx, [])

    while broken_bond_cursors[center_idx] < len(broken_ligands):
        old_ligand_idx = broken_ligands[broken_bond_cursors[center_idx]]
        broken_bond_cursors[center_idx] += 1

        if old_ligand_idx != new_ligand_idx:
            return (
                (idx_to_map[old_ligand_idx], idx_to_map[new_ligand_idx]),
            )

    return ()

#Helper collects carbonyl carbons
def is_carbonyl_carbon(ms, center_idx: int) -> bool:
    center = ms.atoms[center_idx]

    if center.symbol != "C":
        return False

    for bond in center.bonds:
        other = bond.atom2 if bond.atom1 is center else bond.atom1

        if other.symbol == "O" and int(bond.typebondint) == 2:
            return True

    return False

#Helper collects trigonal carbocations

def is_trigonal_carbocation(ms, center_idx: int) -> bool:
    center = ms.atoms[center_idx]

    if center.symbol != "C":
        return False

    if center.charge != 1:
        return False

    neighbors = ms.atom_neighbor_indices(center_idx)

    return len(neighbors) == 3
#Helper detects whether the product can be stereogenic after the move

def product_center_becomes_tetrahedral_stereogenic( #check this helper once again
    product_ms,
    center_idx: int,
) -> bool:
    mol = product_ms.to_rdkit_mol(include_chirality=False)

    possible_centers = product_ms.potential_tetrahedral_chiral_atom_indices(mol)

    return center_idx in possible_centers

#Helper that detects stereogenic events
def is_planar_to_tetrahedral_stereo_event(
    reactant_ms,
    product_ms,
    center_idx: int,
    new_ligand_idx: int,
) -> bool:
    center = reactant_ms.atoms[center_idx]

    if center.symbol != "C":
        return False

    if center.has_tetrahedral_chirality:
        return False

    if new_ligand_idx in reactant_ms.atom_neighbor_indices(center_idx):
        return False

    if not (
        is_carbonyl_carbon(reactant_ms, center_idx)
        or is_trigonal_carbocation(reactant_ms, center_idx)
    ):
        return False

    return product_center_becomes_tetrahedral_stereogenic(
        product_ms,
        center_idx,
    )

def collect_stereo_events(msmi: MechSmiles) -> list[StereoEvent]:
    idx_to_map = {
        atom_idx: map_idx
        for map_idx, atom_idx in msmi.ms.atom_map_dict.items()
    }

    processed_moves = []
    broken_bonds_by_atom: dict[int, list[int]] = defaultdict(list)

    for arrow in msmi.smiles_arrows:
        move = msmi.process_smiles_arrow(arrow, msmi.ms.atom_map_dict)
        processed_moves.append(move)

        if move and move[0] == "i":
            broken_bonds_by_atom[move[1]].append(move[2])
            broken_bonds_by_atom[move[2]].append(move[1])

    try:
        product_ms = msmi.ms.make_move(processed_moves)
    except Exception:
        product_ms = None

    broken_bond_cursors: dict[int, int] = defaultdict(int)
    events: list[StereoEvent] = []

    for move in processed_moves:
        if not move:
            continue

        if move[0] == "a":
            center_idx = move[2]
            new_ligand_idx = move[1]

        elif move[0] == "ba":
            center_idx = move[3]
            new_ligand_idx = move[2]

        else:
            continue

        if center_idx not in idx_to_map:
            continue

        center = msmi.ms.atoms[center_idx]

        if center.has_tetrahedral_chirality:

            if new_ligand_idx not in idx_to_map:
                continue

            ligand_pairs = find_ligand_pairs(
                center_idx=center_idx,
                new_ligand_idx=new_ligand_idx,
                broken_bonds_by_atom=broken_bonds_by_atom,
                broken_bond_cursors=broken_bond_cursors,
                idx_to_map=idx_to_map,
            )

            events.append(
                StereoEvent(
                    center_map=idx_to_map[center_idx],
                    ligand_pairs=ligand_pairs,
                    event_type="tetrahedral_acceptor",
                    mode_options=TETRAHEDRAL_ACCEPTOR_MODES,
                )
            )

        elif product_ms is not None and is_planar_to_tetrahedral_stereo_event(
            reactant_ms=msmi.ms,
            product_ms=product_ms,
            center_idx=center_idx,
            new_ligand_idx=new_ligand_idx,
        ):
            events.append(
                StereoEvent(
                    center_map=idx_to_map[center_idx],
                    ligand_pairs=(),
                    event_type="planar_to_tetrahedral",
                    mode_options=PLANAR_TO_TETRAHEDRAL_MODES,
                )
            )

    return events


def format_stereo_updates(
    events: list[StereoEvent], modes: tuple[str, ...]
) -> str:
    grouped_updates: list[list[object]] = []
    grouped_indices: dict[tuple[int, str], int] = {}

    for event, mode in zip(events, modes, strict=True):
        ligand_pairs = () if mode in {"clear", "unknown", "mix"} else event.ligand_pairs
        key = (event.center_map, mode)

        if key not in grouped_indices:
            grouped_indices[key] = len(grouped_updates)
            grouped_updates.append([event.center_map, mode, []])

        grouped_updates[grouped_indices[key]][2].extend(ligand_pairs)

    return ";".join(
        MechSmiles.format_stereo_update(
            center_map,
            mode,
            tuple(ligand_pairs),
        )
        for center_map, mode, ligand_pairs in grouped_updates
    )


def with_stereo_updates(
    mech_smi: str, events: list[StereoEvent], modes: tuple[str, ...]
) -> str:
    smiles, arrows, _ = split_mech_smi(mech_smi)
    stereo_updates = format_stereo_updates(events, modes)
    return MechSmiles.build_value(smiles, arrows, stereo_updates)


def validate_model_output(
    predicted_mech_smi: str,
    target_smiles: str,
    ignore_stereo_center_maps: set[int] | None = None,
) -> tuple[bool, str]:
    target_counter = canonical_component_counter(
        target_smiles,
        ignore_stereo_center_maps=ignore_stereo_center_maps,
    )
    if target_counter is None:
        return False, "target_canonicalization_failed"

    try:
        predicted_product = MechSmiles(predicted_mech_smi).prod
    except Exception as exc:
        return False, f"product_generation_failed: {type(exc).__name__}: {exc}"

    predicted_counter = canonical_component_counter(
        predicted_product,
        ignore_stereo_center_maps=ignore_stereo_center_maps,
    )
    if predicted_counter is None:
        return False, "predicted_product_canonicalization_failed"

    return predicted_counter == target_counter, ""


def candidate_priority(
    events: list[StereoEvent],
    modes: tuple[str, ...],
) -> tuple[int, ...]:
    priority = []

    for event, mode in zip(events, modes, strict=True):
        if event.event_type == "planar_to_tetrahedral":
            priority.append(0 if mode == "mix" else 1)
        else:
            priority.append(0)

    return tuple(priority)

def resolve_matching_candidates(
    mech_smi: str,
    events: list[StereoEvent],
    matches: dict[str, tuple[str, ...]],
) -> InferenceResult:
    event_types = tuple(event.event_type for event in events)
    ranked = sorted(
        matches.items(),
        key=lambda item: candidate_priority(events, item[1]),
    )

    best_priority = candidate_priority(events, ranked[0][1])
    best_matches = [
        item for item in ranked
        if candidate_priority(events, item[1]) == best_priority
    ]

    if len(best_matches) == 1:
        candidate, modes = best_matches[0]
        return InferenceResult(candidate, "added_" + "_".join(modes), th_event_types=event_types,)

    return InferenceResult(
        mech_smi,
        "ambiguous_matching_mode",
        f"{len(matches)} candidates matched target",
        th_event_types=event_types,
    )

def infer_th_mech_smi(
    mech_smi: str,
    target_smiles: str,
    mode_options: tuple[str, ...] = DEFAULT_MODE_OPTIONS,
) -> InferenceResult:
    if not isinstance(mech_smi, str) or not mech_smi:
        return InferenceResult(str(mech_smi), "invalid_input", "empty mech_smi_min")

    if not isinstance(target_smiles, str) or not target_smiles:
        return InferenceResult(mech_smi, "target_missing", "empty target smiles")

    try:
        msmi = MechSmiles(mech_smi)
        events = collect_stereo_events(msmi)
    except Exception as exc:
        return InferenceResult(
            mech_smi,
            "event_collection_failed",
            f"{type(exc).__name__}: {exc}",
        )

    if not events:
        return InferenceResult(mech_smi, "no_stereo_relevant_event")

    event_types = tuple(event.event_type for event in events)

    ignore_stereo_center_maps = {
        event.center_map
        for event in events
        if event.event_type == "planar_to_tetrahedral"
    }

    matches: dict[str, tuple[str, ...]] = {}
    candidate_errors: list[str] = []

    for modes in product(*(event.mode_options for event in events)):
        candidate = with_stereo_updates(mech_smi, events, modes)
        matches_target, error = validate_model_output(
            candidate,
            target_smiles,
            ignore_stereo_center_maps=ignore_stereo_center_maps,
        )

        if error:
            candidate_errors.append(error)

        if matches_target:
            matches[candidate] = modes

    if not matches:
        error = "; ".join(sorted(set(candidate_errors))[:3])
        return InferenceResult(mech_smi, "no_matching_mode", error, th_event_types=event_types,)

    if len(matches) > 1:
        return resolve_matching_candidates(mech_smi, events, matches)

    candidate, modes = next(iter(matches.items()))
    return InferenceResult(candidate, "added_" + "_".join(modes), th_event_types=event_types,)


def annotate_split_dataset(
    split_name: str,
    dataset: "Dataset",
    mode_options: tuple[str, ...],
    max_rows: int | None = None,
) -> tuple["Dataset", dict[str, Any]]:
    from datasets import Dataset

    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    df = dataset.to_pandas()
    if "split" not in df.columns:
        df["split"] = split_name

    missing_required = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_required:
        raise ValueError(
            f"Split {split_name!r} is missing required columns: {missing_required}"
        )

    targets = build_step_targets(df)
    annotated_values = []
    statuses = []
    errors = []
    target_values = []
    event_type_values = []
    examples = []

    for row_index, row in df.iterrows():
        mech_smi = row["mech_smi_min"]
        target_smiles = targets.get(row_index, "")
        result = infer_th_mech_smi(mech_smi, target_smiles, mode_options)
        event_type_values.append(",".join(result.th_event_types))

        annotated_values.append(result.mech_smi_min_th)
        statuses.append(result.th_status)
        errors.append(result.th_error)
        target_values.append(target_smiles)

        if (
            result.mech_smi_min_th != mech_smi
            and "|TH(" in result.mech_smi_min_th
            and len(examples) < 5
        ):
            examples.append(
                {
                    "rxn_idx": int(row["rxn_idx"]),
                    "step_idx_forward": int(row["step_idx_forward"]),
                    "mech_smi_min": mech_smi,
                    "mech_smi_min_th": result.mech_smi_min_th,
                    "target_smiles_for_validation": target_smiles,
                    "th_status": result.th_status,
                }
            )

    df["mech_smi_min_th"] = annotated_values
    df["th_status"] = statuses
    df["th_error"] = errors
    df["target_smiles_for_validation"] = target_values
    df["th_event_types"] = event_type_values

    output_columns = [column for column in KEEP_COLUMNS if column in df.columns]
    output_columns.extend(ADDED_COLUMNS)
    annotated_df = df[output_columns].copy()

    status_counts = Counter(statuses)
    split_report = {
        "rows": int(len(annotated_df)),
        "rows_with_th_added": int(
            sum(status.startswith("added_") for status in statuses)
        ),
        "status_counts": dict(status_counts),
        "errors": int(sum(bool(error) for error in errors)),
        "examples": examples,
    }
    return Dataset.from_pandas(annotated_df, preserve_index=False), split_report


def annotate_dataset(
    ds: "DatasetDict",
    mode_options: tuple[str, ...] = DEFAULT_MODE_OPTIONS,
    max_rows: int | None = None,
) -> tuple["DatasetDict", dict[str, Any]]:
    from datasets import DatasetDict

    annotated_splits = {}
    split_reports = {}
    global_status_counts: Counter[str] = Counter()
    examples = []
    total_rows = 0
    rows_with_th_added = 0
    error_count = 0

    for split_name, split_dataset in ds.items():
        print(f"Annotating split {split_name} ({len(split_dataset)} rows)")
        annotated_split, split_report = annotate_split_dataset(
            split_name,
            split_dataset,
            mode_options=mode_options,
            max_rows=max_rows,
        )
        annotated_splits[split_name] = annotated_split
        split_reports[split_name] = split_report

        total_rows += split_report["rows"]
        rows_with_th_added += split_report["rows_with_th_added"]
        error_count += split_report["errors"]
        global_status_counts.update(split_report["status_counts"])
        examples.extend(split_report["examples"])

    report = {
        "total_rows": int(total_rows),
        "rows_with_th_added": int(rows_with_th_added),
        "status_counts": dict(global_status_counts),
        "examples": examples[:10],
        "errors": int(error_count),
        "splits": split_reports,
    }
    return DatasetDict(annotated_splits), report


def parse_mode_options(value: str) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in value.split(",") if mode.strip())
    if not modes:
        raise argparse.ArgumentTypeError("--mode_options cannot be empty")

    invalid_modes = sorted(set(modes) - SUPPORTED_MODE_OPTIONS)
    if invalid_modes:
        raise argparse.ArgumentTypeError(
            f"Unsupported mode(s): {invalid_modes}. "
            f"Supported modes are: {sorted(SUPPORTED_MODE_OPTIONS)}"
        )

    return modes


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--push_to_hub", type=parse_bool, default=False)
    parser.add_argument("--hub_repo_id", default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--mode_options",
        type=parse_mode_options,
        default=DEFAULT_MODE_OPTIONS,
        help="Comma-separated stereo modes to test, e.g. invert,clear",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not args.hub_repo_id:
        raise ValueError("--hub_repo_id is required when --push_to_hub true")

    try:
        from datasets import DatasetDict, load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install the project requirements "
            "before running this script."
        ) from exc

    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected a DatasetDict, got {type(ds).__name__}")

    annotated, report = annotate_dataset(
        ds,
        mode_options=args.mode_options,
        max_rows=args.max_rows,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save_to_disk(str(args.output_path))

    report_path = args.output_path / "th_annotation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.push_to_hub:
        annotated.push_to_hub(args.hub_repo_id)

    print(f"Wrote annotated dataset to: {args.output_path}")
    print(f"Wrote report to: {report_path}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Rows with TH added: {report['rows_with_th_added']}")
    print(f"Errors: {report['errors']}")
    print("Status counts:")
    for status, count in Counter(report["status_counts"]).most_common():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
