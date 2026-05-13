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
        "|(6,2);((2,5),5)|TH(2,'mix',((5,6),))"
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
    "'mix'",
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
        ("TH(2,'mix',((5,6),))", (2, "mix", ((5, 6),))),
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
        MechSmiles.format_stereo_update(2, "mix", ((5, 6), (7, 8)))
        == "TH(2,'mix',((5,6),(7,8)))"
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

    mixed = MechSmiles(base + "|TH(2,'mix',((5,6),))").prod
    inverted = MechSmiles(base + "|TH(2,'invert',((5,6),))").prod

    assert "@" in mixed
    assert "@" in inverted
    assert mixed != inverted


def test_TH_clear_removes_chirality():
    msmi = MechSmiles(
        "[F:1][P@:2]([Cl:3])([Br:4])[I:5].[O-:6]"
        "|(6,2);((2,5),5)|TH(2,'clear',())"
    )

    assert "@" not in msmi.prod


def test_TH_retain_and_invert_work_without_ligand_replacement():
    base = "[Cl:1][P@:2]([F:3])([Br:4])[I:5]|(1,2)"

    mixed = MechSmiles(base + "|TH(2,'mix',())").prod
    inverted = MechSmiles(base + "|TH(2,'invert',())").prod

    assert "@" in mixed
    assert "@" in inverted
    assert mixed != inverted


@pytest.mark.parametrize(
    "bad_public_msmi",
    [
        "[Cl:1][P@:2]([F:3])([Br:4])[I:5]|(1,2,'invert')",
        "[Cl:1][P@:2]([F:3])([Br:4])[I:5]|(1,2,'mix')",
        "[CH2:1]=[CH:2][P@:3]([F:4])([Cl:5])[Br:6]|((1,2),3,'invert')",
        "[CH2:1]=[CH:2][P@:3]([F:4])([Cl:5])[Br:6]|((1,2),3,'mix')",
    ],
)
def test_public_arrow_field_rejects_stereo_mode_tokens(bad_public_msmi):
    with pytest.raises(ValueError):
        MechSmiles(bad_public_msmi).prod


def test_move_mechsmiles_serializes_stereo_as_TH_field_not_arrow_mode():
    ms = MechSmiles("[Cl:1][P@:2]([F:3])([Br:4])[I:5]|").ms

    mechsmiles = ms.move_mechsmiles([("a", 0, 1, "invert")])

    assert "|TH(" in mechsmiles
    assert_no_public_stereo_mode_in_arrow_field(mechsmiles)from chrimp.world.mechsmiles import MechSmiles
import pytest
import re
from collections import Counter


def count_atoms(msmi: MechSmiles):
    return Counter([a.GetSymbol() for a in msmi.ms.rdkit_mol.GetAtoms()])


class TestMechSmiles:
    # Additional tests needed
    # - Test that the set of indices used is the set of n first integers

    test_cases = [
        "[H][B-:301]([H])([H])[H:28]|((301, 28), 28)",
        "[H-:301].[H][C:1]([H])([c:2]1[c:3]([H])[c:4]([H])[c:5]([H])[c:6]([H])[c:7]1[H])[Br:101].[H][O:8][c:9]1[c:10]([H])[c:11]([H])[c:12]([O:13][H:17])[c:14]([Br:15])[c:16]1[H].[Na+]|(301, 17);((17, 13), 13)",
        "CCOP(=O)(C[O-:1])OCC.NC1=C2N=C([C:2][Br:3])N(CCC3=CC=CC=C3)C2=NC=N1.[H][H].[Na+]|(1, 2);((2, 3), 3)",
        "[C]=O.C=CC[OH:1].C[S+:2](C)Cl.O=C=O.[Cl-]|(1, 2)",
        "[H][H].[H][C:1]([H])=[O:2].[H+].[H-]|((1, 2), 2)",  # Tricky with explicit Hs
        "Br[B:2](Br)Br.CC(C)N1N=C(C2=CC=C([O:1]C)C=C2)C2=CC=CC(Cl)=C21.O|(1, 2)",  # Explains why we need the double remapping in reduce indices
        "[H][H].[H]C([H])=O.[H+:3].[H-]|",  # Tricky because no move (edge case of hide conditions)
        "CCOP(=O)(C[O-:1])OCC.NC1=C2N=C([C:2][Br:3])N(CCC3=CC=CC=C3)C2=NC=N1.[H][H].[Na+]|(1, 2);((2, 3), 3)",  # Few conditions
    ]

    @pytest.mark.parametrize("init_string", test_cases)
    def test_standardize_is_stable(self, init_string):
        msmi = MechSmiles(init_string)
        msmi.standardize()
        first_std = msmi.value
        msmi.standardize()
        second_std = msmi.value
        assert first_std == second_std, f"Standardize is unstable for: {init_string}"

    @pytest.mark.parametrize("init_string", test_cases)
    def test_atoms_unchanged_after_standardize(self, init_string):
        """Test that atoms remain unchanged after standardization"""
        msmi = MechSmiles(init_string)
        prior_csts = count_atoms(msmi)
        msmi.standardize()
        post_csts = count_atoms(msmi)
        assert prior_csts == post_csts, f"Constants changed for: {init_string}"

    @pytest.mark.parametrize("init_string", test_cases)
    def test_hs_behavior_after_standardize(self, init_string):
        """Test specific hydrogen species behavior: Remove any unmapped explicit H attached to a heavy atom"""
        msmi = MechSmiles(init_string)
        msmi.standardize()
        standardized_str = msmi.smiles

        # Remove the hydrogen species that should remain
        temp_str = re.sub(r"\[H\]\[H\]", "", standardized_str)
        temp_str = re.sub(r"\[H\+\]", "", temp_str)
        temp_str = re.sub(r"\[H-\]", "", temp_str)

        # Check that no other [H] remain
        remaining_explicit_h = re.findall(r"\[H\]", temp_str)
        assert (
            len(remaining_explicit_h) == 0
        ), f"Found {len(remaining_explicit_h)} explicit [H] that should have been removed in '{init_string}': {remaining_explicit_h}"

    @pytest.mark.parametrize("init_string", test_cases)
    def test_atoms_unchanged_after_hise_unhide_cond(self, init_string):
        """Test that atoms remain unchanged after standardization"""
        msmi = MechSmiles(init_string)
        prior_csts = count_atoms(msmi)
        msmi.hide_cond()
        msmi.unhide_cond()
        post_csts = count_atoms(msmi)
        assert prior_csts == post_csts, f"Constants changed for: {init_string}"


# Tests I used to run by hand:
# verbose_tests = True
# My_msmi
# msmi = MechSmiles("[H-:301].[H][C:1]([H])([c:2]1[c:3]([H])[c:4]([H])[c:5]([H])[c:6]([H])[c:7]1[H])[Br:101].[H][O:8][c:9]1[c:10]([H])[c:11]([H])[c:12]([O:13][H:17])[c:14]([Br:15])[c:16]1[H].[Na+]|(301, 17);((17, 13), 13)")
# msmi_ = MechSmiles("[H-:301].[H][C:1]([H])([c:2]1[c:3]([H])[c:4]([H])[c:5]([H])[c:6]([H])[c:7]1[H])[Br:101].[H][O:8][c:9]1[c:10]([H])[c:11]([H])[c:12]([O:13][H:17])[c:14]([Br:15])[c:16]1[H].[Na+]|(301, 17);((17, 13), 13)")

# msmi.standardize(verbose=verbose_tests)
# print(msmi.value)
# print(msmi.standard_value)

# msmi_2 = MechSmiles("[H][B-:302]([H])([H])C#N.[H][C:1]([c:2]1[c:3]([H])[n:4]([H])[c:5]([H])[n:6]1)([N+:7]([C:8]1([H])[C:9]([H])([H])[C:10]([H])([H])[N:11]([C:12]([H])([H])[c:13]2[c:14]([H])[c:15]([H])[c:16]([H])[c:17]([H])[c:18]2[H])[C:19]([H])([H])[C:20]1([H])[H])([H:22])[H:23])[O:101][H].[Na+]|(101, 22);((22, 7), 7)")
# msmi_2.standardize(verbose=verbose_tests)
# print(msmi_2.value)

# msmi_3 = MechSmiles("[CH3:101][O:1][c:2]1[cH:3][cH:4][c:5](-[c:6]2[n:7][n:8]([CH:9]([CH3:10])[CH3:11])[c:12]3[c:13]([Cl:14])[cH:15][cH:16][cH:17][c:18]23)[cH:19][cH:20]1.Br[B:301](Br)[Br:302].[OH2:303]|(1, 301)")
# msmi_3.standardize(verbose=verbose_tests)
# print(msmi_3.value)

## Explains why we need the double remapping in reduce indices
# msmi_4 = MechSmiles("Br[B:2](Br)Br.CC(C)N1N=C(C2=CC=C([O:1]C)C=C2)C2=CC=CC(Cl)=C21.O|(1, 2)")
# msmi_4.standardize(verbose=verbose_tests)
# print(msmi_4.value)

# msmi_5 = MechSmiles("[H][C:1]([H])([C:2]1=[N:3][C:4]2=[C:5]([N:6]([H])[H])[N:7]=[C:8]([H])[N:9]=[C:10]2[N:11]1[C:12]([H])([H])[C:13]([H])([H])[C:14]1=[C:15]([H])[C:16]([H])=[C:17]([H])[C:18]([H])=[C:19]1[H])[Br:101].[H][C:20]([H])([H])[C:21]([H])([H])[O:22][P:23](=[O:24])([C:25]([H])([H])[O-:26])[O:27][C:28]([H])([H])[C:29]([H])([H])[H].[H][H:301].[Na+]|(26, 1);((1, 101), 101)")
# msmi_5.standardize(verbose=verbose_tests)
# print(msmi_5.value)


# msmi_6 = MechSmiles("[H]C([H])([H])[O:302][H].[H][B-:301]([H])([H:29])[H:28].[H][C:1]([H])([H])[O:2][c:3]1[c:4]([H])[c:5]([H])[c:6]([C:7](=[O:8])[C:9]([H])([H])[c:10]2[c:11]([Cl:12])[c:13]([H])[n+:14]([O:15][H])[c:16]([H])[c:17]2[Cl:18])[c:19]2[c:20]1[O:21][C:22]1([C:23]([H])([H])[C:24]([H])([H])[C:25]([H])([H])[C:26]1([H])[H])[O:27]2|((301, 28), 7);((7, 8), 8)")
# msmi_6.standardize(verbose=verbose_tests)
# print(msmi_6.value)

## Check no Hs are disapearing !!
# msmi_7 = MechSmiles("[H][B-:301]([H])([H])[H:28]|((301, 28), 28)")
# msmi_7.standardize(verbose=verbose_tests)
# print(msmi_7.value)

# Same here but Hs are appearing (Not sure chemistry makes a lot of sense here)
# msmi_8 = MechSmiles("[C]=O.C=C1C[OH:1].C[S+:2](C)Cl.O=C=O.[Cl-]|(1, 2)")
# msmi_8.standardize(verbose=verbose_tests)

# exit()

# Hide/show observer
# msmi = MechSmiles("CCOP(=O)(C[O-:1])OCC.NC1=C2N=C([C:2][Br:3])N(CCC3=CC=CC=C3)C2=NC=N1.[H][H].[Na+]|(1, 2);((2, 3), 3)")
# msmi.show()
# msmi.hide_cond()
# print(msmi.value)
# msmi.show()
# msmi.unhide_cond()
# msmi.show()

# Example usage
# my_move = MechSmiles("C[C:2](=[O:3])C.[NH3:1]|(1,2);((2,3), 3)")
##my_move = MechSmiles("C[C:3](=[O:4])C.[BH3-:1][H:2]|((1,2),3);((3,4), 4)")
##my_move = MechSmiles("C[C:2](=[O:3])C.[NH3:1]|")
# my_move.show()
