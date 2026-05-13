import pytest
from collections import Counter
from rdkit import Chem

from chrimp.world.molecule_set import MoleculeSet


def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Counter([a.GetSymbol() for a in mol.GetAtoms()])


class TestMoleculeSet:
    test_cases = [
        ("O=[C-:1][Cl:2]", [("i", 1, 2)])
        # etc...
    ]

    @pytest.mark.parametrize("smiles,move", test_cases)
    def test_atom_count_stable(self, smiles, move):
        ms = MoleculeSet.from_smiles(smiles)
        ms_ = ms.make_move(move)

        count_1 = count_atoms(ms.can_smiles)
        count_2 = count_atoms(ms.mapped_smiles)
        count_3 = count_atoms(ms_.can_smiles)
        count_4 = count_atoms(ms_.mapped_smiles)

        assert count_1 == count_2, f"Atom count changed for: {smiles}"
        assert count_1 == count_3, f"Atom count changed during move {move} on {smiles}"
        assert count_3 == count_4, f"Atom count changed for: {smiles} + move {move}"

    def test_mix_replaces_retain_in_attack_stereo_modes(self):
        assert "mix" in MoleculeSet.attack_stereo_modes
        assert "retain" not in MoleculeSet.attack_stereo_modes

        ms = MoleculeSet.from_smiles("F[P@](Cl)(Br)I")

        with pytest.raises(ValueError):
            ms.make_move(("a", 2, 1, "retain"))

    def test_mix_mode_is_used_for_trigonal_planar_to_tetrahedral_attack(self):
        ms = MoleculeSet.from_smiles("N.P(F)(Cl)Br")

        moves = set(ms.all_legal_attack_moves())
        product = ms.make_move(("a", 0, 1, "mix"))

        assert ("a", 0, 1, "mix") in moves
        assert "@" not in product.can_smiles
        assert not product.atoms[1].has_tetrahedral_chirality

    def test_mix_mode_is_not_used_when_new_center_is_not_stereogenic(self):
        ms = MoleculeSet.from_smiles("N.P(F)(F)Cl")

        moves = set(ms.all_legal_attack_moves())

        assert ("a", 0, 1, "mix") not in moves
        with pytest.raises(ValueError):
            ms.make_move(("a", 0, 1, "mix"))

    def test_mix_mode_is_rejected_for_existing_tetrahedral_acceptor(self):
        ms = MoleculeSet.from_smiles("F[P@](Cl)(Br)I")

        with pytest.raises(ValueError):
            ms.make_move(("a", 2, 1, "mix"))

    def test_move_mechsmiles_encodes_mix_with_empty_ligand_replacements(self):
        ms = MoleculeSet.from_smiles("N.P(F)(Cl)Br")

        mechsmiles = ms.move_mechsmiles([("a", 0, 1, "mix")])

        assert "|TH(" in mechsmiles
        assert ",'mix',()" in mechsmiles
