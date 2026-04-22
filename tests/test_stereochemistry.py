from chrimp.world.molecule_set import MoleculeSet


def get_tetrahedral_atoms(ms):
    return [a for a in ms.atoms if a.has_tetrahedral_chirality]


def test_from_smiles_distinguishes_at_and_atat():
    ms1 = MoleculeSet.from_smiles("F[C@](Cl)(Br)I")
    ms2 = MoleculeSet.from_smiles("F[C@@](Cl)(Br)I")

    stereo1 = get_tetrahedral_atoms(ms1)
    stereo2 = get_tetrahedral_atoms(ms2)

    assert len(stereo1) == 1
    assert len(stereo2) == 1

    a1 = stereo1[0]
    a2 = stereo2[0]

    assert a1.chiral_tag != a2.chiral_tag
    assert set(a1.chiral_neighbors) == set(a2.chiral_neighbors)


def test_can_smiles_preserves_tetrahedral_chirality():
    ms1 = MoleculeSet.from_smiles("F[C@](Cl)(Br)I")
    ms2 = MoleculeSet.from_smiles("F[C@@](Cl)(Br)I")

    assert "@" in ms1.can_smiles
    assert "@" in ms2.can_smiles
    assert ms1.can_smiles != ms2.can_smiles

