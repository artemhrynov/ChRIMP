from chrimp.world.molecule_set import ChrimpAtom, ChrimpBond, MoleculeSet


def get_tetrahedral_atoms(ms):
    return [a for a in ms.atoms if a.has_tetrahedral_chirality]


def replace_ligand(ms, center_idx, old_ligand_idx, new_symbol):
    product = ms.copy()
    center = product.atoms[center_idx]
    old_ligand = product.atoms[old_ligand_idx]
    old_bond = next(
        bond
        for bond in center.bonds
        if bond.atom1 is old_ligand or bond.atom2 is old_ligand
    )

    product.bonds.remove(old_bond)
    old_bond.atom1.bonds.remove(old_bond)
    old_bond.atom2.bonds.remove(old_bond)

    new_ligand = ChrimpAtom(new_symbol, 0, idx=len(product.atoms))
    product.atoms.append(new_ligand)
    product.bonds.append(ChrimpBond(center, new_ligand, 1))
    product.surounding_electrons_calc()
    return product


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


def test_mapped_smiles_preserves_tetrahedral_chirality():
    ms1 = MoleculeSet.from_smiles("F[C@:1](Cl)(Br)I")
    ms2 = MoleculeSet.from_smiles("F[C@@:1](Cl)(Br)I")

    assert "@" in ms1.mapped_smiles
    assert "@" in ms2.mapped_smiles
    assert ":1" in ms1.mapped_smiles
    assert ":1" in ms2.mapped_smiles
    assert ms1.mapped_smiles != ms2.mapped_smiles


def test_can_smiles_preserves_tetrahedral_chirality_with_explicit_hydrogen():
    ms1 = MoleculeSet.from_smiles("F[C@H](Cl)Br")
    ms2 = MoleculeSet.from_smiles("F[C@@H](Cl)Br")

    assert "@" in ms1.can_smiles
    assert "@" in ms2.can_smiles
    assert ms1.can_smiles != ms2.can_smiles


def test_mapped_smiles_preserves_tetrahedral_chirality_with_explicit_hydrogen():
    ms1 = MoleculeSet.from_smiles("F[C@H:1](Cl)Br")
    ms2 = MoleculeSet.from_smiles("F[C@@H:1](Cl)Br")

    assert "@" in ms1.mapped_smiles
    assert "@" in ms2.mapped_smiles
    assert ":1" in ms1.mapped_smiles
    assert ":1" in ms2.mapped_smiles
    assert ms1.mapped_smiles != ms2.mapped_smiles


def test_hydrogen_cleanup_preserves_chiral_bracket_hydrogen():
    ms = MoleculeSet.from_smiles("F[C@H](Cl)Br")

    assert ms.remove_artefact_hydrogens("[C@H](F)(Cl)Br") == "[C@H](F)(Cl)Br"


def test_to_rdkit_mol_preserves_chiral_center_after_ligand_replacement():
    ms1 = replace_ligand(MoleculeSet.from_smiles("F[C@](Cl)(Br)I"), 1, 0, "H")
    ms2 = replace_ligand(MoleculeSet.from_smiles("F[C@@](Cl)(Br)I"), 1, 0, "H")

    assert "@" in ms1.can_smiles
    assert "@" in ms2.can_smiles
    assert ms1.can_smiles != ms2.can_smiles


def test_to_rdkit_mol_drops_chirality_after_replacement_with_duplicate_ligand():
    ms = replace_ligand(MoleculeSet.from_smiles("F[C@](Cl)(Br)I"), 1, 0, "Cl")

    assert "@" not in ms.can_smiles
