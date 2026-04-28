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


def replace_ligands(
    ms,
    center_idx,
    replacements,
    stereo_mode="retain",
    update_chirality=True,
    return_replacement_map=False,
):
    product = ms.copy()
    replacement_map = {}

    for old_ligand_idx, new_symbol in replacements:
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
        replacement_map[old_ligand_idx] = new_ligand.idx

    if update_chirality:
        product.update_tetrahedral_chirality(
            center_idx,
            ligand_replacements=replacement_map,
            stereo_mode=stereo_mode,
        )

    product.surounding_electrons_calc()

    if return_replacement_map:
        return product, replacement_map

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


def test_tracks_tetrahedral_chirality_after_multiple_ligand_replacements():
    ms1 = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [(0, "H"), (2, "F")],
    )
    ms2 = replace_ligands(
        MoleculeSet.from_smiles("F[C@@](Cl)(Br)I"),
        1,
        [(0, "H"), (2, "F")],
    )

    assert "@" in ms1.can_smiles
    assert "@" in ms2.can_smiles
    assert ms1.can_smiles != ms2.can_smiles


def test_can_invert_tetrahedral_chirality_after_multiple_ligand_replacements():
    retained = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [(0, "H"), (2, "F")],
        stereo_mode="retain",
    )
    inverted = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [(0, "H"), (2, "F")],
        stereo_mode="invert",
    )

    assert "@" in retained.can_smiles
    assert "@" in inverted.can_smiles
    assert retained.can_smiles != inverted.can_smiles


def test_drops_chirality_after_multiple_replacements_with_duplicate_ligands():
    ms = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [(0, "H"), (2, "H")],
    )

    assert "@" not in ms.can_smiles
    assert get_tetrahedral_atoms(ms) == []


def test_update_tetrahedral_chirality_handles_all_four_ligand_replacements_with_mapping():
    product, replacement_map = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [
            (0, "Cl"),  # old F  -> new Cl
            (2, "Br"),  # old Cl -> new Br
            (3, "I"),   # old Br -> new I
            (4, "F"),   # old I  -> new F
        ],
        update_chirality=False,
        return_replacement_map=True,
    )

    assert set(replacement_map.keys()) == {0, 2, 3, 4}
    assert len(set(replacement_map.values())) == 4

    result = product.update_tetrahedral_chirality(
        1,
        ligand_replacements=replacement_map,
        stereo_mode="retain",
    )

    assert result is True
    assert "@" in product.can_smiles

    stereo_atoms = get_tetrahedral_atoms(product)
    assert len(stereo_atoms) == 1
    assert set(stereo_atoms[0].chiral_neighbors) == set(replacement_map.values())


def test_update_tetrahedral_chirality_drops_all_four_ligand_replacements_without_mapping():
    product = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [
            (0, "Cl"),
            (2, "Br"),
            (3, "I"),
            (4, "F"),
        ],
        update_chirality=False,
    )

    result = product.update_tetrahedral_chirality(
        1,
        stereo_mode="retain",
    )

    assert result is False
    assert "@" not in product.can_smiles
    assert get_tetrahedral_atoms(product) == []


def test_update_tetrahedral_chirality_can_invert_after_all_four_ligand_replacements():
    retained = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [
            (0, "Cl"),
            (2, "Br"),
            (3, "I"),
            (4, "F"),
        ],
        stereo_mode="retain",
    )

    inverted = replace_ligands(
        MoleculeSet.from_smiles("F[C@](Cl)(Br)I"),
        1,
        [
            (0, "Cl"),
            (2, "Br"),
            (3, "I"),
            (4, "F"),
        ],
        stereo_mode="invert",
    )

    assert "@" in retained.can_smiles
    assert "@" in inverted.can_smiles
    assert retained.can_smiles != inverted.can_smiles


def test_attack_move_on_chiral_acceptor_can_retain_or_invert():
    ms = MoleculeSet.from_smiles("F[P@](Cl)(Br)I")

    retained = ms.make_move(("a", 2, 1, "retain"))
    inverted = ms.make_move(("a", 2, 1, "invert"))

    assert retained.can_smiles == "[F][P@](=[Cl])([Br])[I]"
    assert inverted.can_smiles == "[F][P@@](=[Cl])([Br])[I]"
    assert retained.can_smiles != inverted.can_smiles


def test_legacy_attack_move_on_chiral_acceptor_clears_stereo():
    ms = MoleculeSet.from_smiles("F[P@](Cl)(Br)I")

    product = ms.make_move(("a", 2, 1))

    assert "@" not in product.can_smiles
    assert get_tetrahedral_atoms(product) == []


def test_attack_move_on_nonchiral_acceptor_is_unchanged_by_stereo_annotation():
    ms = MoleculeSet.from_smiles("FP(Cl)(Br)I")

    legacy = ms.make_move(("a", 2, 1))
    retained = ms.make_move(("a", 2, 1, "retain"))
    inverted = ms.make_move(("a", 2, 1, "invert"))

    assert legacy.can_smiles == retained.can_smiles
    assert legacy.can_smiles == inverted.can_smiles


def test_all_legal_attack_moves_enumerate_retain_and_invert_for_chiral_acceptor():
    ms = MoleculeSet.from_smiles("F[P@](Cl)(Br)I")

    moves = {
        move
        for move in ms.all_legal_attack_moves()
        if move[0] == "a" and move[1] == 2 and move[2] == 1
    }

    assert ("a", 2, 1, "retain") in moves
    assert ("a", 2, 1, "invert") in moves


def test_bond_attack_move_on_chiral_acceptor_can_retain_or_invert():
    ms = MoleculeSet.from_smiles("C=C[P@](F)(Cl)Br")

    retained = ms.make_move(("ba", 0, 1, 2, "retain"))
    inverted = ms.make_move(("ba", 0, 1, 2, "invert"))

    assert retained.can_smiles == "[CH2]C=[P@](F)(Cl)Br"
    assert inverted.can_smiles == "[CH2]C=[P@@](F)(Cl)Br"
    assert retained.can_smiles != inverted.can_smiles


def test_all_legal_bond_attack_moves_enumerate_retain_and_invert_for_chiral_acceptor():
    ms = MoleculeSet.from_smiles("C=C[P@](F)(Cl)Br")

    moves = {
        move
        for move in ms.all_legal_bond_attack_moves()
        if move[0] == "ba" and move[1] == 0 and move[2] == 1 and move[3] == 2
    }

    assert ("ba", 0, 1, 2, "retain") in moves
    assert ("ba", 0, 1, 2, "invert") in moves
