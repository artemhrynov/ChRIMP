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
