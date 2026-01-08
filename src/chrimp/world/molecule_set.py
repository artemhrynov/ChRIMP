import re
from typing import Optional, List, Tuple
from itertools import product
from collections import defaultdict, Counter
from copy import deepcopy
import time

from colorama import Fore
from rdkit import Chem

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class ReusedVirtualTSException(Exception):
    """Base class for other custom exceptions"""

    pass


class BondNotFoundError(Exception):
    """Base class for other custom exceptions"""

    pass


class RadicalAtomException(Exception):
    """Base class for other custom exceptions"""

    pass


class ChrimpAtom:
    """
    Class of Atom used in the Chrimp Game
    """

    # 0 would be the null atom
    atomic_num_rare_gas = [0, 2, 10, 18, 36, 54, 86, 118]

    symbol_to_atomic_num = {
        Chem.GetPeriodicTable().GetElementSymbol(i): i for i in range(1, 119)
    }

    def __init__(
        self,
        symbol,
        charge,
        bonds: Optional[List] = None,
        radical=False,
        virtual_ts=False,
        already_vts=False,
        already_donor=False,
        already_acceptor=False,
        idx: Optional[int] = None,
        smiles_explicit=True,
    ):
        self.symbol = symbol
        self.atomic_num = ChrimpAtom.symbol_to_atomic_num[symbol]
        self.charge = charge
        self.virtual_ts = virtual_ts  # An virtual TS in my code is a transition state that violates temporarily physical rules and hence can only be an intermediate in moves
        self.already_used_for_virtual_ts = already_vts
        self.already_used_for_donor = already_donor
        self.already_used_for_acceptor = already_acceptor
        self.radical = radical
        self.idx = idx
        self.smiles_explicit = smiles_explicit  # If the atom is explicit or implicit (in the smiles, not in the molecule set)
        if bonds is None:
            self.bonds = []
        else:
            self.bonds = bonds

        # core and valence electrons
        self.core_valence_calc()

    def __repr__(self):
        return f"ChrimpAtom({self.repr})"

    @property
    def repr(self):
        # representation of the atom
        charge_num_str = f"{abs(self.charge)}" if abs(self.charge) > 1 else ""
        charge_sign_str = "+" if self.charge > 0 else "-"
        charge_str = f"{charge_sign_str}{charge_num_str}" if self.charge != 0 else ""
        idx_str = f" (idx: {self.idx})" if self.idx is not None else ""
        return f"{self.symbol}{charge_str}{idx_str}"

    @property
    def h_idx(self):  # human index (starting at 1)
        return self.idx + 1 if self.idx is not None else None

    def core_valence_calc(self):
        for i in range(len(ChrimpAtom.atomic_num_rare_gas) - 1):
            if self.atomic_num <= 2:  # 1st row (s)
                self.period = 1
                self.core_electrons = 0
                self.max_valence_electrons = 2

            elif 2 < self.atomic_num <= 10:  # 2nd row (s, p)
                self.period = 2
                self.core_electrons = 2
                self.max_valence_electrons = 8

            elif MoleculeSet.default_treat_P_S_Cl_12_electrons and self.atomic_num in [
                15,
                16,
                17,
            ]:
                self.period = 3
                self.core_electrons = 10
                self.max_valence_electrons = 12  # Very rarely exceeds 12, but counter example is perchlorate ClO4-

            elif 10 < self.atomic_num <= 18:  # 3rd row (s, p)
                self.period = 3
                self.core_electrons = 10
                self.max_valence_electrons = 18  # Very rarely exceeds 12, but counter example is perchlorate ClO4-

            elif MoleculeSet.default_treat_Br_I_2nd_period and self.atomic_num in [
                35,
                53,
            ]:  # Treat Br, I as second row
                period_halogens = {35: 4, 53: 5}
                self.period = period_halogens[self.atomic_num]
                self.core_electrons = self.atomic_num - 7
                self.max_valence_electrons = 8

            elif 18 < self.atomic_num <= 30:  # 4th period (4s, 3d)
                self.period = 4
                self.core_electrons = 18
                self.max_valence_electrons = 18

            elif 30 < self.atomic_num <= 36:  # 4th period (4p)
                self.period = 4
                self.core_electrons = 28  # Because 36 - 8e max in sp3 orbital
                self.max_valence_electrons = 18

            elif 36 < self.atomic_num <= 48:  # 5th period (5s, 4d)
                self.period = 5
                self.core_electrons = 36
                self.max_valence_electrons = 18  # In principle, I think no hard quantum mechanics rule avoid hybridyzing like crazy

            elif 48 < self.atomic_num <= 54:  # 5th period (5p)
                self.period = 5
                self.core_electrons = 46  # Because 54 - 8e max in sp3 orbital
                self.max_valence_electrons = 18  # In principle, I think no hard quantum mechanics rule avoid hybridyzing like crazy

            elif 54 < self.atomic_num <= 56:  # 6th period (6s)
                self.period = 6
                self.core_electrons = 54
                self.max_valence_electrons = 18  # In principle, I think no hard quantum mechanics rule avoid hybridyzing like crazy

            else:
                print(f"Atom seen not yet implemented {self.symbol}")
                raise NotImplementedError(
                    f" Atom {self.symbol} never seen, this has to be coded before we continue"
                )

            self.valence_electrons_neutral = self.atomic_num - self.core_electrons

            # if preceding_rg_idx < self.atomic_num <= next_rg_idx:
            #    self.core_electrons = preceding_rg_idx
            #    self.max_valence_electrons = next_rg_idx - preceding_rg_idx
            #    self.valence_electrons_neutral = self.atomic_num - preceding_rg_idx

    def surounding_electrons_calc(self, authorize_radicals=True):
        # Two types of electrons are around an atom:
        # - Shared (in bonds)
        # - Own (in lone pairs and radicals)

        n_electrons_around = self.valence_electrons_neutral - self.charge
        n_shared_electrons_around = 0
        n_pi_bonds_arounds = 0
        unique_pi_list = []  # List of bonds containing pi electrons
        for bond in self.bonds:
            n_shared_electrons_around += bond.typebondint * 2
            n_pi_bonds_arounds += bond.typebondint - 1
            if bond.typebondint > 1:
                unique_pi_list.append(bond)

        n_own_electrons_around = n_electrons_around - (n_shared_electrons_around / 2)
        n_empty_electron_spots = (
            self.max_valence_electrons
            - n_own_electrons_around
            - n_shared_electrons_around
        )

        if not (n_own_electrons_around % 2 == 0 and n_empty_electron_spots % 2 == 0):
            if not authorize_radicals:
                raise RadicalAtomException("Radical atom detected")
            self.radical = True

        # self.n_lone_pairs = n_own_electrons_around // 2
        # self.n_empty_orbitals = n_empty_electron_spots // 2
        self.n_lone_electrons = n_own_electrons_around
        self.n_empty_electron_spots = n_empty_electron_spots
        self.n_pi_bonds = n_pi_bonds_arounds
        self.pi_bonds = unique_pi_list

        virtual_ts = (
            n_own_electrons_around + n_shared_electrons_around
            > self.max_valence_electrons
        )
        if virtual_ts and not self.already_used_for_virtual_ts:
            self.virtual_ts = True
            self.already_used_for_virtual_ts = True
        elif virtual_ts and self.already_used_for_virtual_ts:
            # print(f"Virtual TS already used for {self.repr}")
            raise ReusedVirtualTSException(f"Virtual TS already used for {self.repr}")
        else:
            self.virtual_ts = False


class ChrimpBond:
    """
    Class of Bond used in the Chrimp Game
    """

    bond_int_to_str = {
        1: "single",
        2: "double",
        3: "triple",
        4: "quadruple",
    }

    bond_int_to_smiles = {
        1: "-",
        2: "=",
        3: "#",
        4: "$",
    }

    def __init__(self, atom1, atom2, typebondint):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom1.bonds.append(self)
        self.atom2.bonds.append(self)
        self.at_least_one_heavy_atom = not all(
            [atom1.symbol == "H", atom2.symbol == "H"]
        )
        self.typebondint = typebondint
        self.typebondstr = ChrimpBond.bond_int_to_str[typebondint]
        self.typebondsmi = ChrimpBond.bond_int_to_smiles[typebondint]

    def __repr__(self):
        return f"ChrimpBond({self.atom1.repr}{self.typebondsmi}{self.atom2.repr})"


class MoleculeSet:
    """
    Class of MoleculeSet used in the Chrimp Game
    Improvement on MoleculeSet class in three ways:
        - New move type available: Bond attack
        - New treatment of concertion
        - Stereocenters can be handled properly (Concerted through stereocenter leads to inversion or double-inversion)
    """

    ionization_deprecation_warning_printed = False
    default_hydrogens_implicit = True  # bool
    default_kekulize = True  # bool
    default_treat_Br_I_2nd_period = False  # bool
    default_treat_P_S_Cl_12_electrons = True  # bool
    default_authorize_radicals = True  # bool

    def __init__(
        self, atoms, bonds, chiral: Optional[bool] = None, atom_map_dict: dict = {}
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds

        if chiral is None:
            self.chiral = (
                False  # Is used, in MolBlock, and would be useful to have as an info
            )
        else:
            self.chiral = chiral

        # DON'T FORGET TO REINITIALIZE THESE WHEN COPYING
        self.molblock_ = None
        self.can_smiles_ = None
        self.repr_ = None
        self.atom_map_dict = atom_map_dict

        self.surounding_electrons_calc()

    def clean_smiles_artefacts(self, smiles, rdkit_can=True, verbose=False):
        # This function acts as a SMILES sanitizer, to avoid some simplification of SMILES that don't apply to this exact use case
        smiles_wo_aretacts_h = self.remove_artefact_hydrogens(smiles)
        if rdkit_can:
            smiles_wo_aretacts_h = self.rdkit_canonicalization(smiles_wo_aretacts_h)
        cleaned_smiles = self.put_atoms_in_brackets(smiles_wo_aretacts_h)

        if verbose:
            print(
                f"clean_canonical_smiles: {smiles} -> {smiles_wo_aretacts_h} -> {cleaned_smiles}"
            )
        return cleaned_smiles

    def remove_artefact_hydrogens(self, smiles):
        # This function counteracts the addition that sometimes happens of
        # hydrogens that are neither implicit nor explicit but appear in the SMILES

        # This pattern eliminates any hydrogens not alone in square brackets
        pattern = r"(?<!\[)H\d*"
        cleaned_smiles = re.sub(pattern, "", smiles)
        return cleaned_smiles

    def put_atoms_in_brackets(self, smiles):
        # Put every atom in bracket, to avoid the apparition of implicit hydrogens

        # Match any atom symbol that doesn't need brackets in SMILES notation
        pattern = r"(?<!\[)(Br|B|Cl|C|N|O|P|S|F|I)"

        # Replace each match with the element surrounded by square brackets
        cleaned_smiles = re.sub(pattern, r"[\1]", smiles)

        return cleaned_smiles

    def rdkit_canonicalization(self, smiles, sanitize=False):
        try:
            return Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles, sanitize=sanitize),
                kekuleSmiles=MoleculeSet.default_kekulize,
            )
        except:  # noqa: E722 (Do not use bare except)
            # print(f"Could not canonicalize {smiles}")
            return smiles

    @property
    def can_smiles(self):
        if self.can_smiles_ is None:
            mol = Chem.MolFromMolBlock(self.molblock, removeHs=False, sanitize=False)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            h_can_smiles_ = Chem.MolToSmiles(mol)
            self.can_smiles_ = self.clean_smiles_artefacts(h_can_smiles_)
            if MoleculeSet.default_hydrogens_implicit:
                self.can_smiles_ = self.rdkit_canonicalization(
                    self.can_smiles_, sanitize=True
                )

        return self.can_smiles_

    @property
    def rdkit_mol(self):
        mol = Chem.MolFromMolBlock(self.molblock, removeHs=False, sanitize=False)
        # Since we import with removeHs=False, Hs are explicit and every implicit H is an artifact
        for a in mol.GetAtoms():
            a.SetNoImplicit(True)
        mol.UpdatePropertyCache()
        can_ranks = Chem.CanonicalRankAtoms(mol)
        for i, a in enumerate(mol.GetAtoms()):
            a.SetIntProp("canonical_rank", can_ranks[i])
        return mol

    @property
    def mapped_smiles(self):
        mapped_mol = deepcopy(self.rdkit_mol)
        reverse_atom_map_dict = {v: k for k, v in self.atom_map_dict.items()}
        for atom in mapped_mol.GetAtoms():
            atom.SetAtomMapNum(reverse_atom_map_dict.get(atom.GetIdx(), 0))
        return Chem.MolToSmiles(mapped_mol)

    @property
    def molblock(self):
        if self.molblock_ is None:
            self.molblock_ = self.calc_molblock()
        return self.molblock_

    # Method that takes a smiles and creates a MoleculeSet
    @classmethod
    def from_smiles(cls, smiles) -> "MoleculeSet":
        # If a '*' is in the smiles, we remove it
        if "*" in smiles:
            smiles = smiles.replace("*", "")

        rdkit_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # Kekulize the molecule and add hydrogens
        Chem.Kekulize(rdkit_mol, clearAromaticFlags=True)
        # add a custom feature smiles_explicit being 0 or 1
        for atom in rdkit_mol.GetAtoms():
            atom.SetProp("smiles_explicit", "1")
        rdkit_mol = Chem.AddHs(rdkit_mol)
        for atom in rdkit_mol.GetAtoms():
            # if atom doesn't have the smiles_explicit property, we set it to 0
            if not atom.HasProp("smiles_explicit"):
                atom.SetProp("smiles_explicit", "0")

        already_seen_idx = []
        atoms_list = {}
        connections_list = []
        bonds_list = []
        atom_idx = 0
        atom_map_dict = {}
        for a in rdkit_mol.GetAtoms():
            if a.GetAtomMapNum() != 0:
                atom_map_dict[a.GetAtomMapNum()] = a.GetIdx()
            atoms_list[a.GetIdx()] = ChrimpAtom(
                a.GetSymbol(),
                a.GetFormalCharge(),
                idx=atom_idx,
                smiles_explicit=a.GetProp("smiles_explicit") == "1",
            )
            atom_idx += 1
            already_seen_idx.append(a.GetIdx())
            for b in a.GetBonds():
                indices = [a.GetIdx(), b.GetOtherAtom(a).GetIdx()]
                sorted_indices = sorted(indices)
                connection = (*sorted_indices, b.GetBondTypeAsDouble())
                if connection not in connections_list:
                    connections_list.append(connection)

        for c in connections_list:
            bonds_list.append(ChrimpBond(atoms_list[c[0]], atoms_list[c[1]], c[2]))
        return cls(
            list(atoms_list.values()),
            bonds_list,
            chiral=("@" in smiles),
            atom_map_dict=atom_map_dict,
        )

    def surounding_electrons_calc(self):
        for atom in self.atoms:
            atom.surounding_electrons_calc(
                authorize_radicals=MoleculeSet.default_authorize_radicals
            )

        self.n_virtual_ts = sum([atom.virtual_ts for atom in self.atoms])
        self.virtual_ts_indices = [
            i for i in range(len(self.atoms)) if self.atoms[i].virtual_ts
        ]

    def make_move(self, move) -> "MoleculeSet":
        if not isinstance(move, list):
            move = [move]

        new_ms = self.copy()
        for arrow in move:
            # if any([self.atoms[atom_idx].radical for atom_idx in arrow[1:]]):
            #    raise NotImplementedError("Can create but not yet interact with arrows on radical species")
            new_ms = new_ms.make_one_arrow_move(arrow)
        new_ms.surounding_electrons_calc()
        return new_ms

    def make_one_arrow_move(self, move) -> "MoleculeSet":
        if move[0] == "i":
            new_mol_set = self.make_ionization_move(move)
        elif move[0] == "a":
            new_mol_set = self.make_attack_move(
                move,
                one_e_attack=any(
                    [self.atoms[atom_idx].radical for atom_idx in move[1:]]
                ),
            )
        elif move[0] == "ba":
            new_mol_set = self.make_bond_attack_move(
                move, one_e_attack=self.atoms[move[3]].radical
            )
        elif move[0] == "hv":
            new_mol_set = self.make_homocleavage_move(move)
        else:
            print(f"Move of type {move[0]} not (yet) implemented")
            exit()
        new_mol_set.surounding_electrons_calc()
        return new_mol_set

    def make_bond_attack_move(self, move, one_e_attack: bool = False) -> "MoleculeSet":
        inter_ms = (
            self.make_homocleavage_move(("hv", move[1], move[2]))
            if one_e_attack
            else self.make_ionization_move(("i", move[1], move[2]))
        )
        return inter_ms.make_attack_move(
            ("a", move[2], move[3]), one_e_attack=one_e_attack
        )

    def make_attack_move(self, move, one_e_attack: bool = False) -> "MoleculeSet":
        mol_set_copy = self.copy()
        idx_attacker, idx_electron_acceptor = move[1], move[2]
        attacker, acceptor = (
            mol_set_copy.atoms[idx_attacker],
            mol_set_copy.atoms[idx_electron_acceptor],
        )

        # Check if a bond already exists between the two atoms
        bond_found = False
        for b_idx, bond in enumerate(mol_set_copy.bonds):
            if (
                bond.atom1.idx == idx_attacker
                and bond.atom2.idx == idx_electron_acceptor
            ) or (
                bond.atom1.idx == idx_electron_acceptor
                and bond.atom2.idx == idx_attacker
            ):
                bond_found = True
                bond_idx = b_idx
                break

        if bond_found:
            bond = mol_set_copy.bonds[bond_idx]
            new_bond_order = bond.typebondint + 1
            atom1, atom2 = bond.atom1, bond.atom2

            mol_set_copy.bonds.remove(bond)
            atom1.bonds.remove(bond)
            atom2.bonds.remove(bond)
            mol_set_copy.bonds.append(ChrimpBond(atom1, atom2, new_bond_order))

        else:
            mol_set_copy.bonds.append(ChrimpBond(attacker, acceptor, 1))

        # Update the charges of the atoms
        if not one_e_attack:
            attacker.charge += 1
            acceptor.charge -= 1

            attacker.already_used_for_donor = True
            acceptor.already_used_for_acceptor = True

        return mol_set_copy

    def make_homocleavage_move(self, move) -> "MoleculeSet":
        if len(move) != 3:
            raise ValueError(
                f"Error with homolytic cleavage move of length != 3, move received {move}"
            )

        idx_atom_1, idx_atom_2 = move[1], move[2]
        mol_set_copy = self.copy()

        bond = None
        for bond_ in mol_set_copy.atoms[idx_atom_1].bonds:
            if bond_.atom1.idx == idx_atom_2 or bond_.atom2.idx == idx_atom_2:
                bond = bond_
                break

        if bond is None:
            raise BondNotFoundError(
                f"Could not find a bond between {idx_atom_1} and {idx_atom_2} in molecule {self.can_smiles}"
            )
        assert (
            bond is not None
        ), f"Problem with bond: {bond}, not found bond in between atoms {move[1]} ({self.atoms[move[1]]}) and {move[2]} ({self.atoms[move[2]]})"
        assert isinstance(
            bond, ChrimpBond
        ), (
            f"Problem with bond: {bond}, not of type ChrimpBond"
        )  # Should never be triggered

        if bond.typebondint == 1:  # We break the bond
            bond.atom1.bonds.remove(bond)
            bond.atom2.bonds.remove(bond)
            mol_set_copy.bonds.remove(bond)
        else:
            new_bond_order = bond.typebondint - 1
            atom1, atom2 = bond.atom1, bond.atom2
            atom1.bonds.remove(bond)
            atom2.bonds.remove(bond)
            mol_set_copy.bonds.remove(bond)
            mol_set_copy.bonds.append(ChrimpBond(atom1, atom2, new_bond_order))

        return mol_set_copy

    def make_ionization_move(self, move) -> "MoleculeSet":
        if len(move) > 3:
            if not MoleculeSet.ionization_deprecation_warning_printed:
                print(
                    "Deprecation warning: The bond index was giving bugs, it is now not used anymore"
                )
                MoleculeSet.ionization_deprecation_warning_printed = True

        idx_electron_donor, idx_electron_acceptor = move[1], move[2]

        mol_set_copy = self.copy()

        bond = None
        for bond_ in mol_set_copy.atoms[idx_electron_donor].bonds:
            if (
                bond_.atom1.idx == idx_electron_acceptor
                or bond_.atom2.idx == idx_electron_acceptor
            ):
                bond = bond_
                break

        if bond is None:
            raise BondNotFoundError(
                f"Could not find a bond between {idx_electron_donor} and {idx_electron_acceptor} in molecule {self.can_smiles}"
            )
        assert (
            bond is not None
        ), f"Problem with bond: {bond}, not found bond in between atoms {move[1]} ({self.atoms[move[1]]}) and {move[2]} ({self.atoms[move[2]]})"
        assert isinstance(
            bond, ChrimpBond
        ), (
            f"Problem with bond: {bond}, not of type ChrimpBond"
        )  # Should never be triggered

        donor_atom = mol_set_copy.atoms[idx_electron_donor]
        acceptor_atom = mol_set_copy.atoms[idx_electron_acceptor]

        if bond.typebondint == 1:  # We break the bond
            bond.atom1.bonds.remove(bond)
            bond.atom2.bonds.remove(bond)
            mol_set_copy.bonds.remove(bond)
        else:
            new_bond_order = bond.typebondint - 1
            atom1, atom2 = bond.atom1, bond.atom2
            atom1.bonds.remove(bond)
            atom2.bonds.remove(bond)
            mol_set_copy.bonds.remove(bond)
            mol_set_copy.bonds.append(ChrimpBond(atom1, atom2, new_bond_order))

        donor_atom.charge += 1
        acceptor_atom.charge -= 1
        donor_atom.already_used_for_donor = True
        acceptor_atom.already_used_for_acceptor = True

        return mol_set_copy

    def all_possible_moves(
        self,
        ionization=True,
        attack=True,
        bond_attack=True,
        bond_attack_sigma_allowed=True,
        vts_relax_on_h_allowed=True,
        stop_at_state=None,
        verbose=False,
        overtime_sec=None,
    ):
        start_time = time.time()

        states_to_extend = [(self, [])]
        next_legal_states = defaultdict(
            list
        )  # dictionary of smiles: list of list of arrows (One list of arrows is a move, multiple lists of arrows are multiple moves leading to the same state)

        while states_to_extend:
            if overtime_sec is not None and time.time() - start_time > overtime_sec:
                break

            ms, arrows_history = states_to_extend.pop(0)
            zero_vts, one_vts, more_vts = MoleculeSet.all_possible_one_arrow_states(
                ms,
                arrows_history=arrows_history,
                ionization=ionization,
                attack=attack,
                bond_attack=bond_attack,
                bond_attack_sigma_allowed=bond_attack_sigma_allowed,
                vts_relax_on_h_allowed=vts_relax_on_h_allowed,
                verbose=verbose,
            )

            for next_ms, move in zero_vts.items():
                next_legal_states[next_ms.can_smiles].append(move)
                if stop_at_state is not None and next_ms.can_smiles == stop_at_state:
                    return next_legal_states

            for next_ms, move in one_vts.items():
                states_to_extend.append((next_ms, move))

        return next_legal_states

    @classmethod
    def all_possible_one_arrow_states(
        cls,
        ms: "MoleculeSet",
        arrows_history: List[Tuple] = [],
        ionization=True,
        attack=True,
        bond_attack=True,
        bond_attack_sigma_allowed=True,
        vts_relax_on_h_allowed=True,
        verbose=False,
    ):
        zero_vts = dict()
        one_vts = dict()
        more_vts = dict()
        all_one_arrow_moves = ms.all_possible_one_arrow_moves(
            ionization=ionization,
            attack=attack,
            bond_attack=bond_attack,
            bond_attack_sigma_allowed=bond_attack_sigma_allowed,
            vts_relax_on_h_allowed=vts_relax_on_h_allowed,
        )
        for move in all_one_arrow_moves:
            moves_str = ", ".join([str(tup) for tup in arrows_history + [move]])
            if verbose:
                print(f"Expanding move: {moves_str} (last move: {move})")
            try:
                next_ms = ms.make_move(move)
                if next_ms.n_virtual_ts == 0:
                    zero_vts[next_ms] = arrows_history + [move]
                elif next_ms.n_virtual_ts == 1:
                    one_vts[next_ms] = arrows_history + [move]
                else:
                    more_vts[next_ms] = arrows_history + [move]
            except ReusedVirtualTSException:
                pass

        return zero_vts, one_vts, more_vts

    def all_possible_one_arrow_moves(
        self,
        ionization=True,
        attack=True,
        bond_attack=True,
        bond_attack_sigma_allowed=True,
        vts_relax_on_h_allowed=True,
    ):
        legal_moves = []

        if self.n_virtual_ts == 0:
            virtual_ts_idx = None
        elif self.n_virtual_ts == 1:
            virtual_ts_idx = self.virtual_ts_indices[0]
        else:
            print("Another number of virtual TS than 0 or 1 should not enter here")
            raise ValueError(
                "Another number of virtual TS than 0 or 1 should not enter here"
            )

        # Ionization move (A bond ionizes to any of its ends)
        if ionization:
            legal_moves.extend(
                self.all_legal_ionization_moves(
                    donor_idx=virtual_ts_idx, idx_cannot_accept=self.virtual_ts_indices
                )
            )

        # Attack move (Any atom with a lone pair can attack another atom)
        if attack and virtual_ts_idx is None:  # I think an attack cannot
            legal_moves.extend(self.all_legal_attack_moves(donor_idx=virtual_ts_idx))

        # Bond-attack move (Any atom with pi-bond (for now only pi-bond) can attack another atom)
        if bond_attack:
            legal_moves.extend(
                self.all_legal_bond_attack_moves(
                    donor_idx=virtual_ts_idx,
                    idx_cannot_accept=self.virtual_ts_indices,
                    sigma_allowed=bond_attack_sigma_allowed,
                    can_attack_neutral_h=(
                        vts_relax_on_h_allowed or virtual_ts_idx is None
                    ),
                )
            )

        return legal_moves

    def all_legal_ionization_moves(self, donor_idx=None, idx_cannot_accept=[]):
        # An ionization move has the form: ('i', idx_electron_donor, idx_electron_acceptor)
        i_moves = []
        for bond_idx, bond in enumerate(self.bonds):
            if (
                not bond.atom2.already_used_for_virtual_ts
                and not bond.atom1.already_used_for_donor
                and not bond.atom2.already_used_for_acceptor
            ):
                if (
                    donor_idx is None or donor_idx == bond.atom1.idx
                ) and bond.atom2.idx not in idx_cannot_accept:
                    i_moves.append(("i", bond.atom1.idx, bond.atom2.idx))
            if (
                not bond.atom1.already_used_for_virtual_ts
                and not bond.atom2.already_used_for_donor
                and not bond.atom1.already_used_for_acceptor
            ):
                if (
                    donor_idx is None or donor_idx == bond.atom2.idx
                ) and bond.atom1.idx not in idx_cannot_accept:
                    i_moves.append(("i", bond.atom2.idx, bond.atom1.idx))
        return i_moves

    def all_legal_attack_moves(self, donor_idx=None):
        # An attack move has the form: ('a', idx_attacker, idx_electron_acceptor)
        a_moves = []
        # TODO if accepting end is chiral, one need to account for that and add two moves (inversion and double-inversion)
        for atom, other_atom in product(self.atoms, self.atoms):
            if donor_idx is None or donor_idx == atom.idx:
                if (
                    atom != other_atom
                    and atom.n_lone_electrons > 0
                    and not atom.already_used_for_donor
                    and not other_atom.already_used_for_acceptor
                ):
                    a_moves.append(("a", atom.idx, other_atom.idx))
        return a_moves

    def all_legal_bond_attack_moves(
        self,
        donor_idx=None,
        can_attack_neutral_h=True,
        sigma_allowed=True,
        idx_cannot_accept=[],
    ):
        # A bond attack move has the form: ('ba', idx_electron_donor, idx_attacker, idx_electron_acceptor)
        ba_moves = []
        for atom in self.atoms:
            # Tag any bond linked to a VTS or pi-bonds
            bonds_that_can_attack = []

            for bond in atom.bonds:
                if sigma_allowed:
                    bonds_that_can_attack.append(bond)
                else:
                    # Pi bonds
                    if bond.typebondint > 1:
                        bonds_that_can_attack.append(bond)
                    # Sigma linked to a VTS
                    # We might want to remove them from heuristic or only allow them to attack atoms adjacent to the attacker as heuristic
                    elif bond.atom1.virtual_ts or bond.atom2.virtual_ts:
                        bonds_that_can_attack.append(bond)

            for bond in bonds_that_can_attack:
                donor, attacker = (
                    (bond.atom1, bond.atom2)
                    if bond.atom2.idx == atom.idx
                    else (bond.atom2, bond.atom1)
                )
                if donor_idx is None or donor_idx == donor.idx:
                    for acceptor in [
                        a
                        for a in self.atoms
                        if (a.idx not in [attacker.idx, donor.idx])
                    ]:
                        if (
                            can_attack_neutral_h
                            or acceptor.symbol != "H"
                            or acceptor.charge != 0
                        ):
                            if not (
                                donor.already_used_for_donor
                                or acceptor.already_used_for_acceptor
                                or attacker.already_used_for_acceptor
                                or attacker.already_used_for_donor
                            ):
                                ba_moves.append(
                                    ("ba", donor.idx, attacker.idx, acceptor.idx)
                                )
        return ba_moves

    def calc_molblock(self, vts_is_uranium=False):
        molblock = "\n"  # Title line
        molblock += "ChRIMP RL environment\n"  # Program line
        molblock += "\n"  # Comment line
        molblock += f"{len(self.atoms):3.0f}{len(self.bonds):3.0f}  0  0  {int(self.chiral)}  0  0  0  0  0999 V2000\n"  # Counts line

        charges_list = []
        for atom in self.atoms:
            molblock += f"    0.0000    0.0000    0.0000 {atom.symbol if ((not vts_is_uranium) or (not atom.virtual_ts)) else 'U':<3} 0  0  0  0  0  0  0  0  0  0  0  0\n"  # Check all these 0's are ok
            if atom.charge != 0:
                charges_list.append(f"{atom.h_idx:4.0f}{atom.charge:4.0f}")
        for bond in self.bonds:
            # Bond stereo: page 46 of https://discover.3ds.com/sites/default/files/2020-08/biovia_ctfileformats_2020.pdf
            bond_stereo = 0
            molblock += f"{bond.atom1.h_idx:3.0f}{bond.atom2.h_idx:3.0f}{bond.typebondint:3.0f}{bond_stereo:3.0f}\n"

        charges_block = f"M  CHG{len(charges_list):3.0f}"
        for idx_charge in charges_list:
            charges_block += idx_charge
        molblock += f"{charges_block}\n" if len(charges_list) > 0 else ""
        molblock += "M  END\n"
        return molblock

    def calc_repr(self):
        if self.n_virtual_ts == 0:
            self.repr_ = self.can_smiles
        elif self.n_virtual_ts == 1:
            modif_mol_block = self.calc_molblock(vts_is_uranium=True)
            vts = [atom for atom in self.atoms if atom.virtual_ts][0]
            charge_num_str = f"{abs(vts.charge)}" if abs(vts.charge) > 1 else ""
            charge_sign_str = "+" if vts.charge > 0 else "-"
            charge_str = f"{charge_num_str}{charge_sign_str}" if vts.charge != 0 else ""
            rdkit_mol = Chem.MolFromMolBlock(
                modif_mol_block, removeHs=False, sanitize=False
            )
            Chem.Kekulize(rdkit_mol, clearAromaticFlags=True)
            rdkit_mol = Chem.AddHs(rdkit_mol)
            rdkit_smi = Chem.MolToSmiles(rdkit_mol)
            # rdkit_smi = Chem.MolToSmiles(Chem.MolFromMolBlock(modif_mol_block,removeHs=False,sanitize=False))
            rdkit_smi = self.clean_smiles_artefacts(rdkit_smi)

            pattern = r"\[U.*?\]"
            # We don't have to add neighboring hydrogens with we use Chem.AddHs above
            # neigboring_hs = 0
            # for b in vts.bonds:
            #    if (b.atom1.symbol == "H" and b.atom2.idx == vts.idx) or (b.atom2.symbol == "H" and b.atom1.idx == vts.idx):
            #        neigboring_hs += 1
            # hydrogens_str = f"H{neigboring_hs}" if neigboring_hs > 0 else ""
            self.repr_ = re.sub(pattern, f"[*{vts.symbol}{charge_str}]", rdkit_smi)

        else:
            self.repr_ = None

    @property
    def repr(self):
        if self.repr_ is None:
            self.calc_repr()
        return self.repr_

    def __repr__(self):
        if self.n_virtual_ts in [0, 1]:
            return f"MoleculeSet({self.repr})"
        else:
            return f"MoleculeSet({Fore.RED}multiple virtual TS, not implemented yet{Fore.RESET})"

    def copy(self) -> "MoleculeSet":
        # We want the copy to reinitialize the features that have been calculated
        ms_copy = deepcopy(self)
        ms_copy.molblock_ = None
        ms_copy.can_smiles_ = None
        ms_copy.repr_ = None
        ms_copy.atom_map_dict = self.atom_map_dict.copy()

        return ms_copy

    def check_step_legality(
        self,
        next_smiles_or_mol_set,
        only_heuristic=False,
        show_svg=False,
        verbose=False,
        vverbose=False,
        overtime_sec=None,
    ) -> Tuple[bool, List, str]:  # returns bool of legality and the move
        """
        Check if there exists a legal step in the game leading to the next molecule set

        Args:
            next_smiles_or_mol_set: The next smiles or MoleculeSet to check
            only_heuristic: If True, only check the heuristic moves (cannot propagate a virtual TS by bond-attack on a neutral H)
            show_svg: If True, show the SVG of the move
            verbose: If True, print verbose information
            vverbose: If True, print very verbose information
            overtime_sec: If not None, the maximum time to spend on the check

        Returns:
            bool: True if the step is legal, False otherwise
            List: The move leading to the next molecule set
            str: The reason why the step is legal or not
        """

        if isinstance(next_smiles_or_mol_set, str):
            next_ms = MoleculeSet.from_smiles(next_smiles_or_mol_set)
        else:
            next_ms = next_smiles_or_mol_set

        if vverbose:
            verbose = True

        if not self.check_a_priori_step_legality(next_ms):
            if verbose:
                print("A priori check failed")
            return False, [], "A priori failed"

        # If we have some overlapping molecules on both sides, we can know they will not interact with the arrows
        reac_smi = self.can_smiles
        counter_reac = Counter(reac_smi.split("."))
        prod_smi = next_ms.can_smiles
        counter_prod = Counter(prod_smi.split("."))

        counter_overlap = counter_reac & counter_prod

        # If the overlap is empty, we try very_heuristic (Avoids infinite recursion)
        if len(counter_overlap) > 0:
            print(f"Overlap: {counter_overlap}")
            non_overlapping_reac_smi = ".".join(counter_reac - counter_overlap)
            non_overlapping_prod_smi = ".".join(counter_prod - counter_overlap)

            # IF the prod after removing the overlap is empty, every transformation satisfies the condition
            # If the reac after removing the overlap is empty, we aldready have everything (Should never happen without prod also empty -> no reaction)

            if len(non_overlapping_reac_smi) == 0 or len(non_overlapping_prod_smi) == 0:
                return True, [], "Trivial transformation"

            print(
                f"Overlap detected, retrying with non-overlapping:\n{non_overlapping_reac_smi}>{'.'.join(counter_overlap)}>{non_overlapping_prod_smi}"
            )

            non_overlapping_ms = MoleculeSet.from_smiles(non_overlapping_reac_smi)
            non_overlapping_next_ms = MoleculeSet.from_smiles(non_overlapping_prod_smi)

            legal, move, reason = non_overlapping_ms.check_step_legality(
                non_overlapping_next_ms,
                only_heuristic=only_heuristic,
                show_svg=False,
                verbose=verbose,
                vverbose=vverbose,
                overtime_sec=overtime_sec,
            )

            if legal:  # If the very heuristic check worked, we can return the result
                if show_svg and legal:
                    non_overlapping_ms.show_move_svg(move)

            return legal, move, reason

        # Else we need to check there exists a 1 move path from self to next_ms
        # First, we'll check with the heuristic that you cannot relax a virtual TS by bond-attack on a Hydrogen (Very rare move that use quite a lot of calculation)
        # We could also add the heuristic that exact molecule match on rhs and lhs probably don't interact in the arrow pushing (careful, we have to do a mapping before removing any molecule for the new idx to correspond to old idx)
        next_ms_can_smiles = next_ms.can_smiles
        all_states_and_moves_heuristic = self.all_possible_moves(
            ionization=True,
            attack=True,
            bond_attack=True,
            vts_relax_on_h_allowed=False,
            stop_at_state=next_ms_can_smiles,
            verbose=vverbose,
            overtime_sec=None
            if overtime_sec is None
            else (overtime_sec if only_heuristic else overtime_sec / 2),
        )
        for item in all_states_and_moves_heuristic.items():
            if item[0] == next_ms_can_smiles:
                if show_svg:
                    self.show_move_svg(item[1][0])
                if verbose:
                    print("Heuristic worked")
                return True, item[1][0], "Heuristic"

        if not only_heuristic:
            # If the heuristic didn't work, we'll do the full calculation
            all_states_and_moves = self.all_possible_moves(
                ionization=True,
                attack=True,
                bond_attack=True,
                vts_relax_on_h_allowed=True,
                stop_at_state=next_ms_can_smiles,
                verbose=vverbose,
                overtime_sec=None if overtime_sec is None else overtime_sec / 2,
            )
            for item in all_states_and_moves.items():
                if item[0] == next_ms_can_smiles:
                    if show_svg:
                        self.show_move_svg(item[1][0])
                    if verbose:
                        print("Heuristic didn't work but the full calculation did")
                    return True, item[1][0], "Extended"

        # if overtime_sec is not None and time.time() - check_step_legality_start_time > overtime_sec:
        #    if verbose:
        #        print("Overtime")
        #    return False, [], f"Overtime ({overtime_sec:.0f}s)"

        if verbose:
            print("Everything ran but the move was not found")
        return (
            False,
            [],
            ("Not found" + (" (only heuristic)" if only_heuristic else "")),
        )

    def proton_missing(
        self, next_ms
    ) -> Tuple[
        bool, int
    ]:  # Bool yes or no, int number of protons missing on the right: positive if missing, negative if too much
        curr_atom_dict, curr_total_charge = self.atom_dictionary()
        next_atom_dict, next_total_charge = next_ms.atom_dictionary()

        if (
            abs(diff_H := (curr_atom_dict.get("H", 0) - next_atom_dict.get("H", 0)))
            == 0
        ):
            return False, 0
        if abs(diff_charge := (curr_total_charge - next_total_charge)) == 0:
            return False, 0
        # Verify we are missing protons and not hydrides
        if diff_charge * diff_H < 0:
            return False, 0

        everything_same_expept_H = True

        for symbol, curr_count in curr_atom_dict.items():
            if symbol != "H" and curr_count != next_atom_dict[symbol]:
                everything_same_expept_H = False
                break

        if everything_same_expept_H:
            return True, diff_H

    def check_a_priori_step_legality(self, next_ms):
        # Some set of quick checks that can be done before even starting expensive calculations

        # Compute the sets of charges and atoms
        curr_atom_dict, curr_total_charge = self.atom_dictionary()
        next_atom_dict, next_total_charge = next_ms.atom_dictionary()

        # Prior check, a step is imaginable only is next_atom_dict is a subset of curr_atom_dict or is they are equal AND the total charge is the same

        if not set(next_atom_dict.keys()).issubset(
            set(curr_atom_dict.keys())
        ):  # Alchemy
            return False

        no_atom_lost = True
        for atom, curr_count in curr_atom_dict.items():
            if curr_count < next_atom_dict[atom]:
                return False
            elif curr_count > next_atom_dict[atom]:
                no_atom_lost = False

        # print(f"Charge check: {curr_total_charge} == {next_total_charge}: {curr_total_charge == next_total_charge}")
        # print(f"Atom dicts are:\n{curr_atom_dict}\n{next_atom_dict}")

        if no_atom_lost and curr_total_charge != next_total_charge:
            return False

        else:
            return True

    def show_move_svg(
        self,
        move,
        save_path=None,
        invisible_circles=False,
        hv_icon=None,
        return_svg=False,
        implicit_hs=True,
    ):
        from chrimp.visualization.arrows_on_mols import arrows_on_mol

        if not isinstance(move, list):
            move = [move]

        if implicit_hs:
            explicit_idx = [
                i
                for i in range(len(self.atoms))
                if self.atoms[i].smiles_explicit
                and (
                    self.atoms[i].symbol != "H"
                    or not any([b.at_least_one_heavy_atom for b in self.atoms[i].bonds])
                )
            ]
        else:
            explicit_idx = [
                i for i in range(len(self.atoms)) if self.atoms[i].smiles_explicit
            ]

        radical_arrows = any(
            [any([self.atoms[i].radical for i in arrow[1:]]) for arrow in move]
        )

        if radical_arrows:
            print(f"Radical species: {[a for a in self.atoms if a.radical]}")

        # None if return_svg is False
        return arrows_on_mol(
            self.molblock,
            arrows=move,
            save_path=save_path,
            explicit_idx=explicit_idx,
            invisible_circles=invisible_circles,
            hv_icon=hv_icon,
            return_svg=return_svg,
            radical_arrows=radical_arrows,
        )

    def move_mechsmiles(self, move):
        rdkit_mol = Chem.MolFromMolBlock(self.molblock, removeHs=False, sanitize=False)
        Chem.Kekulize(rdkit_mol, clearAromaticFlags=True)

        arrows = []
        for m in move:
            if m[0] == "a":
                arrows.append((m[1], m[2]))
            elif m[0] == "i":
                arrows.append(((m[1], m[2]), m[2]))
            elif m[0] == "ba":
                arrows.append(((m[1], m[2]), m[3]))
            else:
                raise NotImplementedError(
                    "In 'move_mechsmiles', only moves of types 'a', 'i' and 'ba' are considered for now"
                )

        for i, atom in enumerate(rdkit_mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)

        # Increment all indices in the moves by 1
        all_used_indices = set()

        def increment_tuple(tup):
            if isinstance(tup, int):
                all_used_indices.add(tup + 1)
                return tup + 1
            elif isinstance(tup, tuple):
                return tuple(increment_tuple(i) for i in tup)

        pre_final_arrows = [increment_tuple(arrow) for arrow in arrows]
        lower_int_convert = {k: (i + 1) for i, k in enumerate(all_used_indices)}

        def convert_tuple(tup, dic):
            if isinstance(tup, int):
                return dic[tup]
            elif isinstance(tup, tuple):
                return tuple(convert_tuple(i, dic) for i in tup)

        final_arrows = [
            convert_tuple(arrow, lower_int_convert) for arrow in pre_final_arrows
        ]

        for idx_atom, atom in enumerate(rdkit_mol.GetAtoms()):
            if atom.GetAtomMapNum() not in all_used_indices:
                atom.SetAtomMapNum(0)
            else:
                atom.SetAtomMapNum(lower_int_convert[atom.GetAtomMapNum()])

        rwmol = Chem.RWMol(rdkit_mol)
        indices_to_remove = []
        for atom in rwmol.GetAtoms():
            if atom.GetSymbol() == "H" and atom.GetAtomMapNum() == 0:
                if len(atom.GetBonds()) > 0:  # if it has at least one bond
                    indices_to_remove.append(atom.GetIdx())

        # Remove from highest to lowest index
        for idx in sorted(indices_to_remove, reverse=True):
            rwmol.RemoveAtom(idx)
        rdkit_mol = rwmol.GetMol()

        mech_smiles_string = f"{Chem.MolToSmiles(rdkit_mol, kekuleSmiles=True)}|{';'.join([str(a) for a in final_arrows])}"
        # print(f"{Fore.GREEN}MechSmiles string: {mech_smiles_string}{Fore.RESET}")
        return mech_smiles_string

    def atom_dictionary(self):
        # Return a dictionary with the number of each type of atom as well as the total charge.
        # This dictionary should always remain constant before and after a legal move, so this
        # function serves as an assertion test.
        atom_dict = defaultdict(int)
        total_charge = 0

        for a in self.atoms:
            atom_dict[a.symbol] += 1
            total_charge += a.charge

        return atom_dict, total_charge

    # Comparison methods
    def tag_attacks_on_bonds(self, move) -> list[bool]:
        """
        This method is only used as a comparison against the format of mech-USPTO-31k
        """
        if not isinstance(move, list):
            move = [move]

        new_mol_set = self.copy()
        attacks_on_bond = []
        for arrow in move:
            # if any([self.atoms[atom_idx].radical for atom_idx in arrow[1:]]):
            #    raise NotImplementedError("Can create but not yet interact with arrows on radical species")

            if arrow[0] == "i":
                is_attack_on_bond = False  # not an attack
                new_mol_set = new_mol_set.make_ionization_move(arrow)

            elif arrow[0] == "a":
                idx_attacker, idx_electron_acceptor = arrow[1], arrow[2]

                # Check if a bond already exists between the two atoms
                bond_found = False
                for b_idx, bond in enumerate(new_mol_set.bonds):
                    if (
                        bond.atom1.idx == idx_attacker
                        and bond.atom2.idx == idx_electron_acceptor
                    ) or (
                        bond.atom1.idx == idx_electron_acceptor
                        and bond.atom2.idx == idx_attacker
                    ):
                        bond_found = True
                        break

                is_attack_on_bond = bond_found

                new_mol_set = new_mol_set.make_attack_move(
                    arrow,
                    one_e_attack=any(
                        [self.atoms[atom_idx].radical for atom_idx in arrow[1:]]
                    ),
                )

            elif arrow[0] == "ba":
                idx_attacker, idx_electron_acceptor = arrow[2], arrow[3]

                # Check if a bond already exists between the two atoms
                bond_found = False
                for b_idx, bond in enumerate(new_mol_set.bonds):
                    if (
                        bond.atom1.idx == idx_attacker
                        and bond.atom2.idx == idx_electron_acceptor
                    ) or (
                        bond.atom1.idx == idx_electron_acceptor
                        and bond.atom2.idx == idx_attacker
                    ):
                        bond_found = True
                        break

                is_attack_on_bond = bond_found
                new_mol_set = new_mol_set.make_bond_attack_move(
                    arrow, one_e_attack=self.atoms[arrow[3]].radical
                )
            else:
                print(f"Move of type {arrow[0]} not (yet) implemented")
                exit()
            attacks_on_bond.append(is_attack_on_bond)

        return attacks_on_bond


if __name__ == "__main__":
    import time

    def all_legal_moves(smi):
        start_time = time.time()
        ms = MoleculeSet.from_smiles(smi)
        all_states_and_moves = ms.all_possible_moves(
            ionization=True, attack=True, bond_attack=True, verbose=True
        )

        sum_states = len(all_states_and_moves)
        sum_moves = sum([len(moves) for moves in all_states_and_moves.values()])
        for state, moves in all_states_and_moves.items():
            print(f"{state}: e.g. {moves[0]} ({len(moves)} equivalent paths)")
        print(f"Number of unique legal next states: {sum_states}")
        print(f"Total number of legal moves: {sum_moves}")
        print(f"Time taken: {time.time()-start_time:.2f}s")

    def check_legal_jump(smi, smi_goal):
        ms = MoleculeSet.from_smiles(smi)
        print(
            f"Check step legality from {smi} to {smi_goal}:\n{ms.check_step_legality(smi_goal, show_svg=True, verbose=True)}"
        )

    # smiles = "[H][H]"
    # smiles = "CCO"
    # smiles = "C=CC=C.C=C"
    # smiles = "C=C.[H+]"
    # smiles="C=O.[F-]"
    # smiles = "[CH-]1CCCC[CH+]1"
    smiles = "C1CCCCC1=O.N"
    # smiles = "C1C(C)CCCC1=O.N"
    # smiles = "CC([H])=O.CO.CO.[H+]" # acetal

    # all_legal_moves(smiles)

    # Little additional test (smi_1 and smi_2 are the same molecule, only differ by placement of the double bond inside the pyridine ring)
    # smi_1 = "C[O-].NC1C=CN=C(Cl)C=1"
    smi_2 = "C[O-].NC1=CC=NC(Cl)=C1"
    smi_goal = "NC1C=C[N-]C(OC)(Cl)C=1"

    # Show the intermediate ~SMILES~
    # ms2 = MoleculeSet.from_smiles(smi_2)
    # ms3 = ms2.make_move([('a', 1, 7)])
    # print(ms3.repr)

    # check_legal_jump(smi_1, smi_goal)

    check_legal_jump(smi_2, smi_goal)

    exit()
