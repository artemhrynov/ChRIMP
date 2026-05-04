from collections import Counter
import re

import pytest

from chrimp.world.mechsmiles import MechSmiles


LEGACY_TEST_CASES = [
    "[H][B-:301]([H])([H])[H:28]|((301, 28), 28)",
    (
        "[H-:301].[H][C:1]([H])([c:2]1[c:3]([H])[c:4]([H])"
        "[c:5]([H])[c:6]([H])[c:7]1[H])[Br:101].[H][O:8]"
        "[c:9]1[c:10]([H])[c:11]([H])[c:12]([O:13][H:17])"
        "[c:14]([Br:15])[c:16]1[H].[Na+]"
        "|(301, 17);((17, 13), 13)"
    ),
    (
        "CCOP(=O)(C[O-:1])OCC.NC1=C2N=C([C:2][Br:3])"
        "N(CCC3=CC=CC=C3)C2=NC=N1.[H][H].[Na+]"
        "|(1, 2);((2, 3), 3)"
    ),
    "[C]=O.C=CC[OH:1].C[S+:2](C)Cl.O=C=O.[Cl-]|(1, 2)",
    "[H][H].[H][C:1]([H])=[O:2].[H+].[H-]|((1, 2), 2)",
    (
        "Br[B:2](Br)Br.CC(C)N1N=C(C2=CC=C([O:1]C)C=C2)"
        "C2=CC=CC(Cl)=C21.O|(1, 2)"
    ),
    "[H][H].[H]C([H])=O.[H+:3].[H-]|",
    (
        "CCOP(=O)(C[O-:1])OCC.NC1=C2N=C([C:2][Br:3])"
        "N(CCC3=CC=CC=C3)C2=NC=N1.[H][H].[Na+]"
        "|(1, 2);((2, 3), 3)"
    ),
]


CANONICAL_STEREO_EXAMPLES = [
    (
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)|TH(2,'retain',((5,6),))"
    ),
    (
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)|TH(2,'invert',((5,6),))"
    ),
    (
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)|TH(2,'clear',())"
    ),
    (
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)|TH(2,'unknown',())"
    ),
]


PUBLIC_STEREO_MODE_TOKENS = [
    "'retain'",
    "'invert'",
    "'clear'",
    "'unknown'",
]


def count_atoms(msmi: MechSmiles):
    return Counter(atom.GetSymbol() for atom in msmi.ms.rdkit_mol.GetAtoms())


def arrow_field(msmi_value: str) -> str:
    parts = msmi_value.split("|")
    return parts[1] if len(parts) > 1 else ""


def stereo_field(msmi_value: str) -> str:
    parts = msmi_value.split("|")
    return parts[2] if len(parts) > 2 else ""


def assert_no_public_stereo_mode_in_arrow_field(msmi_value: str):
    arrows = arrow_field(msmi_value)

    for mode_token in PUBLIC_STEREO_MODE_TOKENS:
        assert mode_token not in arrows, (
            "Public MechSMILES should not encode tetrahedral stereo mode "
            f"inside the arrow field: {msmi_value}"
        )


@pytest.mark.parametrize("init_string", LEGACY_TEST_CASES)
def test_standardize_is_stable_for_legacy_mechsmiles(init_string):
    msmi = MechSmiles(init_string)

    msmi.standardize()
    first_std = msmi.value

    msmi.standardize()
    second_std = msmi.value

    assert first_std == second_std, f"Standardize is unstable for: {init_string}"


@pytest.mark.parametrize("init_string", LEGACY_TEST_CASES)
def test_atoms_unchanged_after_standardize_for_legacy_mechsmiles(init_string):
    msmi = MechSmiles(init_string)

    prior_atoms = count_atoms(msmi)
    msmi.standardize()
    post_atoms = count_atoms(msmi)

    assert prior_atoms == post_atoms, f"Atom counts changed for: {init_string}"


@pytest.mark.parametrize("init_string", LEGACY_TEST_CASES)
def test_hydrogen_cleanup_after_standardize_for_legacy_mechsmiles(init_string):
    msmi = MechSmiles(init_string)
    msmi.standardize()

    standardized_smiles = msmi.smiles

    # Keep hydrogen species that are allowed to remain as isolated species.
    tmp = re.sub(r"\[H\]\[H\]", "", standardized_smiles)
    tmp = re.sub(r"\[H\+\]", "", tmp)
    tmp = re.sub(r"\[H-\]", "", tmp)

    remaining_explicit_h = re.findall(r"\[H\]", tmp)

    assert remaining_explicit_h == [], (
        f"Found explicit [H] atoms that should have been removed in "
        f"{init_string}: {remaining_explicit_h}"
    )


@pytest.mark.parametrize("init_string", LEGACY_TEST_CASES)
def test_atoms_unchanged_after_hide_unhide_conditions(init_string):
    msmi = MechSmiles(init_string)

    prior_atoms = count_atoms(msmi)

    msmi.hide_cond()
    msmi.unhide_cond()

    post_atoms = count_atoms(msmi)

    assert prior_atoms == post_atoms, f"Atom counts changed for: {init_string}"


def test_process_smiles_arrow_supports_plain_atom_attack():
    msmi = MechSmiles("[Cl:1][P@:2]([F:3])([Br:4])[I:5]|(1,2)")

    move = msmi.process_smiles_arrow("(1,2)", msmi.ms.atom_map_dict)

    assert move == ("a", 0, 1)


def test_process_smiles_arrow_supports_plain_bond_attack():
    msmi = MechSmiles("[CH2:1]=[CH:2][P@:3]([F:4])([Cl:5])[Br:6]|((1,2),3)")

    move = msmi.process_smiles_arrow("((1,2),3)", msmi.ms.atom_map_dict)

    assert move == ("ba", 0, 1, 2)


@pytest.mark.parametrize(
    "stereo_string, expected",
    [
        ("TH(2,'invert',((5,6),))", (2, "invert", ((5, 6),))),
        ("TH(2,'retain',((5,6),))", (2, "retain", ((5, 6),))),
        ("TH(2,'clear',())", (2, "clear", ())),
        ("TH(2,'unknown',())", (2, "unknown", ())),
    ],
)
def test_parse_stereo_update_accepts_canonical_TH_format(stereo_string, expected):
    assert MechSmiles.parse_stereo_update(stereo_string) == expected


@pytest.mark.parametrize(
    "bad_stereo_string",
    [
        "TH(2,'invert')",
        "TH('2','invert',((5,6),))",
        "TH(2,'wrong',((5,6),))",
        "TH(2,'clear',((5,6),))",
        "TH(2,'unknown',((5,6),))",
        "TH(2,'invert',[5,6])",
        "TH(2,'invert',((5,),))",
    ],
)
def test_parse_stereo_update_rejects_invalid_TH_format(bad_stereo_string):
    with pytest.raises(ValueError):
        MechSmiles.parse_stereo_update(bad_stereo_string)


def test_format_stereo_update_uses_canonical_TH_format():
    assert (
        MechSmiles.format_stereo_update(2, "invert", ((5, 6),))
        == "TH(2,'invert',((5,6),))"
    )

    assert (
        MechSmiles.format_stereo_update(2, "retain", ((5, 6), (7, 8)))
        == "TH(2,'retain',((5,6),(7,8)))"
    )

    assert MechSmiles.format_stereo_update(2, "clear", ()) == "TH(2,'clear',())"


def test_collect_stereo_update_indices():
    indices = MechSmiles.collect_stereo_update_indices("TH(20,'invert',((50,60),))")

    assert indices == {"20", "50", "60"}


def test_remap_stereo_update():
    reactive_indices_dict = {
        "20": 2,
        "50": 5,
        "60": 6,
    }

    remapped = MechSmiles.remap_stereo_update(
        "TH(20,'invert',((50,60),))",
        reactive_indices_dict,
    )

    assert remapped == "TH(2,'invert',((5,6),))"


def test_process_stereo_update_maps_ligand_replacement():
    msmi = MechSmiles(
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)|TH(2,'invert',((5,6),))"
    )

    update = msmi.process_stereo_update(
        "TH(2,'invert',((5,6),))",
        msmi.ms.atom_map_dict,
    )

    assert update.center_idx == msmi.ms.atom_map_dict[2]
    assert update.stereo_mode == "invert"
    assert update.ligand_replacements == {
        msmi.ms.atom_map_dict[5]: msmi.ms.atom_map_dict[6]
    }
    assert update.original_chiral_neighbors == (
        msmi.ms.atoms[update.center_idx].chiral_neighbors
    )


@pytest.mark.parametrize("init_string", CANONICAL_STEREO_EXAMPLES)
def test_canonical_stereo_mechsmiles_has_no_stereo_mode_in_arrow_field(init_string):
    msmi = MechSmiles(init_string)

    assert_no_public_stereo_mode_in_arrow_field(msmi.value)
    assert "TH(" in stereo_field(msmi.value)


@pytest.mark.parametrize("init_string", CANONICAL_STEREO_EXAMPLES)
def test_canonical_stereo_mechsmiles_standardize_is_stable(init_string):
    msmi = MechSmiles(init_string)

    msmi.standardize()
    first_std = msmi.value

    assert_no_public_stereo_mode_in_arrow_field(first_std)
    assert "TH(" in stereo_field(first_std)

    msmi.standardize()
    second_std = msmi.value

    assert first_std == second_std


def test_stereo_update_standardize_remaps_indices():
    msmi = MechSmiles(
        "[F:10][P@:20]([Cl:30])([Br:40])[I:50].[O-:60]"
        "|(60,20);((20,50),50)|TH(20,'invert',((50,60),))"
    )

    msmi.standardize()

    center_map, stereo_mode, ligand_pairs = MechSmiles.parse_stereo_update(
        msmi.stereo_update_strings[0]
    )

    mapped_indices = set(map(int, re.findall(r":(\d+)]", msmi.smiles)))

    assert len(msmi.value.split("|")) == 3
    assert stereo_mode == "invert"
    assert {center_map, *ligand_pairs[0]}.issubset(mapped_indices)
    assert_no_public_stereo_mode_in_arrow_field(msmi.value)


def test_legacy_mechsmiles_without_TH_clears_chirality_on_chiral_acceptor():
    msmi = MechSmiles(
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)"
    )

    assert "@" not in msmi.prod


def test_TH_retain_and_invert_preserve_different_chiral_products_after_ligand_change():
    base = (
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)"
    )

    retained = MechSmiles(base + "|TH(2,'retain',((5,6),))").prod
    inverted = MechSmiles(base + "|TH(2,'invert',((5,6),))").prod

    assert "@" in retained
    assert "@" in inverted
    assert retained != inverted


def test_TH_clear_removes_chirality():
    msmi = MechSmiles(
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)|TH(2,'clear',())"
    )

    assert "@" not in msmi.prod


def test_TH_retain_and_invert_work_without_ligand_replacement():
    base = "[Cl:1][P@:2]([F:3])([Br:4])[I:5]|(1,2)"

    retained = MechSmiles(base + "|TH(2,'retain',())").prod
    inverted = MechSmiles(base + "|TH(2,'invert',())").prod

    assert "@" in retained
    assert "@" in inverted
    assert retained != inverted


@pytest.mark.parametrize(
    "bad_public_msmi",
    [
        "[Cl:1][P@:2]([F:3])([Br:4])[I:5]|(1,2,'invert')",
        "[Cl:1][P@:2]([F:3])([Br:4])[I:5]|(1,2,'retain')",
        "[CH2:1]=[CH:2][P@:3]([F:4])([Cl:5])[Br:6]|((1,2),3,'invert')",
        "[CH2:1]=[CH:2][P@:3]([F:4])([Cl:5])[Br:6]|((1,2),3,'retain')",
    ],
)
def test_public_arrow_field_rejects_stereo_mode_tokens(bad_public_msmi):
    with pytest.raises(ValueError):
        MechSmiles(bad_public_msmi).prod


def test_move_mechsmiles_serializes_stereo_as_TH_field_not_arrow_mode():
    ms = MechSmiles("[Cl:1][P@:2]([F:3])([Br:4])[I:5]|").ms

    mechsmiles = ms.move_mechsmiles([("a", 0, 1, "invert")])

    assert "|TH(" in mechsmiles
    assert_no_public_stereo_mode_in_arrow_field(mechsmiles)