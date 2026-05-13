"""Add canonical TH(...) stereo updates to the preprocessed mech-USPTO-31k CSV.

The existing preprocessed dataset stores elementary steps as:

    mapped_atoms|electron_movement

This script keeps the same rows and columns, preserves the mapped stereochemical
SMILES text (including @ and @@), and appends a third field when a step attacks a
pre-existing tetrahedral stereocenter:

    mapped_atoms|electron_movement|TH(center,'mode',ligand_pairs)

The TH mode is inferred by comparing the product generated with candidate
stereo modes against the next elementary-step reactant, or against the original
USPTO product for the last elementary step.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from chrimp.world.mechsmiles import MechSmiles


DEFAULT_INPUT = Path("data/preprocessed/mech_uspto_31k.csv")
DEFAULT_ORIGINAL = Path("data/uspto-31k/original/mech-USPTO-31k.csv")
DEFAULT_OUTPUT = Path("data/preprocessed/mech_uspto_31k_with_th.csv")


@dataclass(frozen=True)
class StereoEvent:
    center_map: int
    ligand_pairs: tuple[tuple[int, int], ...]


def split_mech_smi(mech_smi: str) -> tuple[str, str, str]:
    parts = mech_smi.split("|", 2)
    smiles = parts[0]
    arrows = parts[1] if len(parts) > 1 else ""
    stereo = parts[2] if len(parts) > 2 else ""
    return smiles, arrows, stereo


def canonical_unmapped_smiles(smiles: str, uncharge: bool = False) -> str | None:
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

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    if uncharge:
        try:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        except Exception:
            return None

    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def canonical_component_counter(
    smiles: str, uncharge: bool = False
) -> Counter[str] | None:
    canonical = canonical_unmapped_smiles(smiles, uncharge=uncharge)
    if canonical is None:
        return None
    if canonical == "":
        return Counter()
    return Counter(canonical.split("."))


def counter_contains(container: Counter[str], contained: Counter[str]) -> bool:
    return not (contained - container)


def original_products_by_source(original_path: Path) -> dict[str, str]:
    raw = pd.read_csv(original_path)
    raw[["reac_smiles", "cond_smiles", "prod_smiles"]] = raw[
        "updated_reaction"
    ].str.split(">", expand=True)

    no_duplicates = raw.drop_duplicates(
        subset=["reac_smiles", "prod_smiles", "mechanistic_label"]
    )
    reactions = no_duplicates[
        no_duplicates.apply(lambda row: row["reac_smiles"] != row["prod_smiles"], axis=1)
    ].copy()
    reactions["source"] = [f"rxn_{idx}" for idx in range(len(reactions))]

    return dict(zip(reactions["source"], reactions["prod_smiles"], strict=False))


def step_targets(df: pd.DataFrame, source_products: dict[str, str]) -> dict[int, str]:
    targets: dict[int, str] = {}

    for _, group in df.groupby("rxn_idx", sort=False):
        ordered = group.sort_values("step_idx_retro", ascending=False)
        ordered_indices = list(ordered.index)

        for position, row_index in enumerate(ordered_indices):
            if position + 1 < len(ordered_indices):
                next_mech_smi = df.at[ordered_indices[position + 1], "mech_smi"]
                targets[row_index] = split_mech_smi(next_mech_smi)[0]
            else:
                source = df.at[row_index, "source"]
                targets[row_index] = source_products.get(source, "")

    return targets


def collect_stereo_events(msmi: MechSmiles) -> list[StereoEvent]:
    idx_to_map = {atom_idx: map_idx for map_idx, atom_idx in msmi.ms.atom_map_dict.items()}
    processed_moves = []
    broken_bonds_by_atom: dict[int, list[int]] = defaultdict(list)

    for arrow in msmi.smiles_arrows:
        move = msmi.process_smiles_arrow(arrow, msmi.ms.atom_map_dict)
        processed_moves.append(move)

        if move and move[0] == "i":
            broken_bonds_by_atom[move[1]].append(move[2])
            broken_bonds_by_atom[move[2]].append(move[1])

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

        if not msmi.ms.atoms[center_idx].has_tetrahedral_chirality:
            continue

        ligand_pairs: tuple[tuple[int, int], ...] = ()
        broken_ligands = broken_bonds_by_atom.get(center_idx, [])

        while broken_bond_cursors[center_idx] < len(broken_ligands):
            old_ligand_idx = broken_ligands[broken_bond_cursors[center_idx]]
            broken_bond_cursors[center_idx] += 1
            if old_ligand_idx != new_ligand_idx:
                ligand_pairs = (
                    (idx_to_map[old_ligand_idx], idx_to_map[new_ligand_idx]),
                )
                break

        events.append(
            StereoEvent(
                center_map=idx_to_map[center_idx],
                ligand_pairs=ligand_pairs,
            )
        )

    return events


def format_stereo_updates(
    events: list[StereoEvent], modes: tuple[str, ...]
) -> str:
    grouped_updates: list[list[object]] = []
    grouped_indices: dict[tuple[int, str], int] = {}

    for event, mode in zip(events, modes, strict=True):
        ligand_pairs = () if mode in {"clear", "unknown"} else event.ligand_pairs
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


def candidate_matches(candidate_mech_smi: str, target_counter: Counter[str]) -> bool:
    try:
        candidate_product = MechSmiles(candidate_mech_smi).prod
    except Exception:
        return False

    candidate_counter = canonical_component_counter(candidate_product)
    if candidate_counter is None:
        return False

    return candidate_counter == target_counter


def infer_th_mech_smi(mech_smi: str, target_smiles: str) -> tuple[str, str]:
    if "@" not in mech_smi:
        return mech_smi, "no_input_stereo"

    target_counter = canonical_component_counter(target_smiles)
    if target_counter is None:
        return mech_smi, "target_canonicalization_failed"

    try:
        msmi = MechSmiles(mech_smi)
        events = collect_stereo_events(msmi)
    except Exception:
        return mech_smi, "event_collection_failed"

    if not events:
        return mech_smi, "no_chiral_acceptor"

    mode_options = ("retain", "invert", "clear")
    matches: list[tuple[str, tuple[str, ...]]] = []

    for modes in product(mode_options, repeat=len(events)):
        candidate = with_stereo_updates(mech_smi, events, modes)
        if candidate_matches(candidate, target_counter):
            matches.append((candidate, modes))

    if len(matches) == 0:
        return mech_smi, "no_matching_mode"

    if len(matches) > 1:
        return mech_smi, "ambiguous_matching_mode"

    candidate, modes = matches[0]
    return candidate, "added_" + "_".join(modes)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    source_products = original_products_by_source(args.original)
    targets = step_targets(df, source_products)

    statuses: Counter[str] = Counter()
    transformed = []

    for row_index, mech_smi in df["mech_smi"].items():
        new_mech_smi, status = infer_th_mech_smi(mech_smi, targets.get(row_index, ""))
        transformed.append(new_mech_smi)
        statuses[status] += 1

    df["mech_smi"] = transformed
    df.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Rows with TH: {df['mech_smi'].str.contains(r'\|TH\(').sum()}")
    print(f"Rows preserving @/@@: {df['mech_smi'].str.contains('@', regex=False).sum()}")
    print("Statuses:")
    for status, count in statuses.most_common():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
