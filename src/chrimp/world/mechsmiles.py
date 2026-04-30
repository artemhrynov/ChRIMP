import re
import ast
from dataclasses import dataclass
from rdkit import Chem
from colorama import Fore
from collections import Counter

from chrimp.world.utils import quick_canonicalize
from chrimp.visualization.arrows_on_mols import filter_hydrogens
from chrimp.world.molecule_set import MoleculeSet, RadicalAtomException

try:
    from chrimp.visualization.mechsmiles_visualizer import MechSmilesVisualizer
except ModuleNotFoundError:
    MechSmilesVisualizer = None


class NotInitiatingAttackError(Exception):
    """
    Exception raised when the drawn arrow is not initiating an attack.
    """

    pass


class MechSmilesContextError(Exception):
    """
    Exception raised when the context of a MechSMILES is incoherent with its value
    """

    pass


class MechSmilesInitError(Exception):
    """
    Exception raised when the context of a MechSMILES cannot initialize
    """

    pass


@dataclass
class StereoUpdate:
    center_idx: int
    stereo_mode: str
    ligand_replacements: dict
    original_chiral_tag: object | None = None
    original_chiral_neighbors: tuple | None = None


class MechSmiles:
    """
    A class to represent a mechanistic SMILES string.
    The idea is to map at least the atoms involved by the arrows, and then give the arrows respecting this mapping.

    example:
    # Ammonia attacking an acetone would be the following: (Or an equivalent permutation of indices)

    "C[C:2](=[O:3])C.[NH3:1]|(1,2);((2,3), 3)" # Here the tuples are start of the arrow, end of the arrow

    If a bond attacks, we will give we will give this bond as a tuple, the arrow will become ((source, inter), sink)
    example:
    # NaBH4 reducing an acetone, would be the following: (or an equivalent permutation of indices)

    "C[C:3](=[O:4])C.[NaH3-:1][H:2]|((1,2),3);((3,4), 4)"
    """

    visualizer = MechSmilesVisualizer() if MechSmilesVisualizer is not None else None
    warning_dummy_already_printed = False

    default_hide_observers = False

    @staticmethod
    def build_value(smiles, arrows_string="", stereo_string=""):
        value = f"{smiles}|{arrows_string}"
        if stereo_string:
            value += f"|{stereo_string}"
        return value

    @staticmethod
    def remap_stereo_update(stereo_string, reactive_indices_dict):
        center_map, stereo_mode, ligand_pairs = MechSmiles.parse_stereo_update(
            stereo_string
        )

        new_center = int(reactive_indices_dict[str(center_map)])

        new_pairs = tuple(
            (
                int(reactive_indices_dict[str(old_ligand)]),
                int(reactive_indices_dict[str(new_ligand)]),
            )
            for old_ligand, new_ligand in ligand_pairs
        )

        return MechSmiles.format_stereo_update(new_center, stereo_mode, new_pairs)

    @staticmethod
    def format_stereo_update(center_map, stereo_mode, ligand_pairs):
        if ligand_pairs:
            pairs_string = (
                "("
                + ",".join(
                    f"({old_ligand},{new_ligand})"
                    for old_ligand, new_ligand in ligand_pairs
                )
                + ("," if len(ligand_pairs) == 1 else "")
                + ")"
            )
        else:
            pairs_string = "()"

        return f"TH({center_map},{stereo_mode!r},{pairs_string})"

    @staticmethod
    def collect_stereo_update_indices(stereo_string):
        center_map, _, ligand_pairs = MechSmiles.parse_stereo_update(stereo_string)

        indices = {str(center_map)}

        for old_ligand, new_ligand in ligand_pairs:
            indices.add(str(old_ligand))
            indices.add(str(new_ligand))

        return indices

    @staticmethod
    def parse_stereo_update(stereo_string):
        if not stereo_string.startswith("TH(") or not stereo_string.endswith(")"):
            raise ValueError(f"Unknown stereo update: {stereo_string}")

        payload = stereo_string[2:]

        try:
            stereo_update = ast.literal_eval(payload)
        except Exception as e:
            raise ValueError(f"Invalid stereo update format: {stereo_string}") from e

        if not isinstance(stereo_update, tuple) or len(stereo_update) != 3:
            raise ValueError(
                "Tetrahedral stereo update must have format "
                "TH(center,'mode',((old_ligand,new_ligand),))"
            )

        center_map, stereo_mode, ligand_pairs = stereo_update

        if type(center_map) is not int:
            raise ValueError(f"Stereo update center must be an integer: {center_map!r}")

        if stereo_mode not in MoleculeSet.attack_stereo_modes:
            raise ValueError(f"Invalid stereo mode: {stereo_mode}")

        if not isinstance(ligand_pairs, tuple):
            raise ValueError("Stereo update ligand replacements must be a tuple")

        if stereo_mode in {"clear", "unknown"} and ligand_pairs:
            raise ValueError(
                f"Stereo mode {stereo_mode!r} must use empty ligand replacements: ()"
            )

        for pair in ligand_pairs:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(type(ligand) is not int for ligand in pair)
            ):
                raise ValueError(
                    "Stereo update ligand replacements must be "
                    "(old_ligand,new_ligand) integer pairs"
                )

        return center_map, stereo_mode, ligand_pairs

    @staticmethod
    def parse_arrow_tuple(arrow_smiles):
        if isinstance(arrow_smiles, str):
            arrow_smiles = re.sub(r"hv", r'"hv"', arrow_smiles)
            return eval(arrow_smiles)
        raise ValueError(
            f"Arrow must be a string, instead is: {type(arrow_smiles)} with value {arrow_smiles}"
        )

    @staticmethod
    def collect_arrow_indices(tup):
        if isinstance(tup, int):
            return {str(tup)}
        if isinstance(tup, tuple):
            indices = set()
            for item in tup:
                indices.update(MechSmiles.collect_arrow_indices(item))
            return indices
        return set()

    @staticmethod
    def remap_arrow_tuple(tup, reactive_indices_dict):
        if isinstance(tup, int):
            return int(reactive_indices_dict[str(tup)])
        if isinstance(tup, str):
            return tup
        if isinstance(tup, tuple):
            return tuple(
                MechSmiles.remap_arrow_tuple(i, reactive_indices_dict) for i in tup
            )
        return tup

    def __init__(
        self,
        value: str,
        context: str | None = None,
    ):
        # Initialize the MechSmiles instance
        self.init_everything_from_value(value, context=context)

    def init_everything_from_value(self, value, conds=None, context=None):
        self.value = value
        value_split = value.split("|", 2)
        self.smiles = value_split[0]
        try:
            self.ms = MoleculeSet.from_smiles(self.smiles)
        except RadicalAtomException as e:
            raise RadicalAtomException(e)
        except:  # noqa: E722 (Do not use bare except)
            raise MechSmilesInitError(f"Invalid smiles {self.smiles}")
        if conds is None:
            self.conds = [].copy()  # List of SMILES
        else:
            self.conds = conds.copy()
        self.context = context

        if context is not None:
            # We update the conditions
            already_acounted_species_value = Counter(
                [quick_canonicalize(x) for x in self.ms.can_smiles.split(".")]
            )
            already_acounted_species_cond = Counter(
                [quick_canonicalize(c) for c in self.conds]
            )
            all_context = Counter([quick_canonicalize(x) for x in context.split(".")])

            if not set(already_acounted_species_value.keys()).issubset(
                set(all_context.keys())
            ):
                # print(f"{set(already_acounted_species_value.keys())=}")
                # print(f"{set(all_context.keys())=}")
                raise MechSmilesContextError(
                    f"Context incorrect, doesn't contain the reacting species\n{already_acounted_species_value=}\n{all_context=}"
                )

            difference = (
                all_context
                - already_acounted_species_value
                - already_acounted_species_cond
            )
            self.conds.extend(list(difference.elements()))

        if len(value_split) > 1:
            self.all_arrows_string = value_split[1]
        else:
            self.all_arrows_string = ""

        if len(value_split) > 2:
            self.all_stereo_string = value_split[2]
        else:
            self.all_stereo_string = ""

        if len(self.all_stereo_string) > 0:
            self.stereo_update_strings = self.all_stereo_string.split(";")
        else:
            self.stereo_update_strings = []

        if len(self.all_arrows_string) > 0:
            self.smiles_arrows = self.all_arrows_string.split(";")
        else:
            self.smiles_arrows = []

        self._prod = None
        self._ms_prod = None

    def show_reac(self, **kwargs):
        if MechSmiles.visualizer is None:
            raise ModuleNotFoundError(
                "Visualization dependencies are not installed for MechSmiles."
            )
        return MechSmiles.visualizer.show_reac(self, **kwargs)

    def show_prod(self, **kwargs):
        if MechSmiles.visualizer is None:
            raise ModuleNotFoundError(
                "Visualization dependencies are not installed for MechSmiles."
            )
        return MechSmiles.visualizer.show_prod(self, **kwargs)

    def show_cond(self, **kwargs):
        if MechSmiles.visualizer is None:
            raise ModuleNotFoundError(
                "Visualization dependencies are not installed for MechSmiles."
            )
        return MechSmiles.visualizer.show_cond(self, **kwargs)

    def show(self, **kwargs):
        if MechSmiles.visualizer is None:
            raise ModuleNotFoundError(
                "Visualization dependencies are not installed for MechSmiles."
            )
        return MechSmiles.visualizer.show(self, **kwargs)

    def hide_cond(self, unmap_non_reactive_atoms=True):
        # Will be considered conditions every species that doesn't have mapping,
        # By default we will then unmap everything that doesn't participate in
        # at least one arrow

        if unmap_non_reactive_atoms:
            self.unmap_nonreactive_atoms()

        regex = re.compile(r":(\d+)]")

        species = self.smiles.split(".")
        new_smiles = ".".join([s for s in species if regex.search(s)])
        self.init_everything_from_value(
            self.build_value(
                new_smiles,
                self.all_arrows_string,
                self.all_stereo_string,
            ),
            conds=[s for s in species if not regex.search(s)],
        )

    def unhide_cond(self):
        new_smiles = ".".join(([self.smiles] if self.smiles != "" else []) + self.conds)
        self.init_everything_from_value(
            self.build_value(
                new_smiles,
                self.all_arrows_string,
                self.all_stereo_string,
            ),
            conds=[],
        )

    def standardize(self, verbose=False):
        if verbose:
            print(f"Initial:        {self.value}")
        self.unmap_nonreactive_atoms()
        if verbose:
            print(f"After unmap:    {self.value}")
        self.kekulize()
        if verbose:
            print(f"After kekulize: {self.value}")
        self.minimize_indices()
        if verbose:
            print(f"Final:          {self.value}")

    def kekulize(self):
        mol = Chem.MolFromSmiles(self.smiles, sanitize=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
        )

        ## first we'll try to kekulize with the Hs
        ## (important to do in that order to avoid heteroatom KekulizeException)
        ## https://github.com/rdkit/rdkit/wiki/FrequentlyAskedQuestions#cant-kekulize-mol
        new_smiles = Chem.MolToSmiles(mol)

        # then, remove unmapped Hs
        new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)
        Chem.SanitizeMol(
            new_mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
        )
        mapped_or_isolated_idx = [
            a.GetIdx()
            for a in new_mol.GetAtoms()
            if (
                a.GetSymbol() == "H"
                and (
                    a.GetAtomMapNum() > 0
                    or all([b.GetOtherAtom(a).GetSymbol() == "H" for b in a.GetBonds()])
                )
            )
        ]
        new_new_mol, _ = filter_hydrogens(new_mol, [], mapped_or_isolated_idx)

        self.init_everything_from_value(
            self.build_value(
                Chem.MolToSmiles(new_new_mol),
                self.all_arrows_string,
                self.all_stereo_string,
            )
        )

    def process_stereo_update(self, stereo_string, atom_map_dict):
        center_map, stereo_mode, ligand_pairs = self.parse_stereo_update(stereo_string)

        center_idx = atom_map_dict[center_map]

        ligand_replacements = {
            atom_map_dict[old_ligand_map]: atom_map_dict[new_ligand_map]
            for old_ligand_map, new_ligand_map in ligand_pairs
        }

        return StereoUpdate(
            center_idx=center_idx,
            stereo_mode=stereo_mode,
            ligand_replacements=ligand_replacements,
            original_chiral_tag=self.ms.atoms[center_idx].chiral_tag,
            original_chiral_neighbors=self.ms.atoms[center_idx].chiral_neighbors,
        )

    def unmap_nonreactive_atoms(self):
        """
        Remove the mapping of all atoms that do not participate in the arrow-pushes
        """
        reactive_indices = set()

        for arrow in self.smiles_arrows:
            reactive_indices.update(
                self.collect_arrow_indices(self.parse_arrow_tuple(arrow))
            )

        for stereo_string in self.stereo_update_strings:
            reactive_indices.update(
                self.collect_stereo_update_indices(stereo_string)
            )
        # get all indices in SMILES
        indices_smiles = set(re.findall(r":(\d+)]", self.smiles))

        non_reactive_indices = indices_smiles - reactive_indices
        final_smiles = self.smiles

        for idx in non_reactive_indices:
            # Replace the :\d+] pattern by a simple closing square bracket ]
            final_smiles = re.sub(f":{idx}]", "]", final_smiles)

        self.init_everything_from_value(
            self.build_value(
                final_smiles,
                self.all_arrows_string,
                self.all_stereo_string,
            )
        )

    def minimize_indices(self):
        """
        Reassign the indices of the MechSMILES in a way to:
        - Have the minimal distinct integer (1, 2, ... n)
        - Have them appear in increasing order in the SMILES part
        """
        # Get all the mapped indices and reorder them
        reactive_indices = re.findall(r":(\d+)]", self.value)

        # This case is an example of why we need to do it with 2 dicts, to first free all needed indices, and then remap.
        # Br[B:2](Br)Br.CC(C)N1N=C(C2=CC=C([O:1]C)C=C2)C2=CC=CC(Cl)=C21.O|(1, 2)

        max_reactive_index = (
            max([int(x) for x in reactive_indices]) if len(reactive_indices) > 0 else 0
        )
        dict_remap_1 = {
            str(idx): str(max_reactive_index + i + 1)
            for i, idx in enumerate(reactive_indices)
        }
        dict_remap_2 = {
            str(max_reactive_index + i + 1): str(i + 1)
            for i, _ in enumerate(reactive_indices)
        }

        final_smiles = self.smiles

        tmp_smiles_arrows = [
            str(self.remap_arrow_tuple(self.parse_arrow_tuple(tup_str), dict_remap_1))
            for tup_str in self.smiles_arrows
        ]
        for old_idx, new_idx in dict_remap_1.items():
            final_smiles = re.sub(f":{old_idx}]", f":{new_idx}]", final_smiles)

        tmp_stereo_updates = [
            self.remap_stereo_update(stereo_string, dict_remap_1)
            for stereo_string in self.stereo_update_strings
        ]

        all_arrows_string = ";".join(
            str(x)
            for x in [
                self.remap_arrow_tuple(self.parse_arrow_tuple(tup_str), dict_remap_2)
                for tup_str in tmp_smiles_arrows
            ]
        )
        for old_idx, new_idx in dict_remap_2.items():
            final_smiles = re.sub(f":{old_idx}]", f":{new_idx}]", final_smiles)

        all_stereo_string = ";".join(
            self.remap_stereo_update(stereo_string, dict_remap_2)
            for stereo_string in tmp_stereo_updates
        )

        self.init_everything_from_value(
            self.build_value(final_smiles, all_arrows_string, all_stereo_string)
        )

    def check_validity(self):
        """
        Check if the mechanistic SMILES string is valid.
        This is a placeholder for actual validation logic.
        """
        # Checking every bond in 'i' and 'ba' exist, and all atoms in 'a' have at least one lone pair, 'a' must initiate, otherwise, it's a rush warning.
        if not MechSmiles.warning_dummy_already_printed:
            print(
                f"{Fore.YELLOW}Careful, validity not implemented yet, this is a dummy test{Fore.RESET}"
            )
            MechSmiles.warning_dummy_already_printed = True
        return True

    @property
    def standard_value(self):
        self.standardize()
        return self.value

    def apply_stereo_update(self, mol_set, stereo_update):
        center = mol_set.atoms[stereo_update.center_idx]
        if stereo_update.original_chiral_tag is not None:
            center.chiral_tag = stereo_update.original_chiral_tag
        if stereo_update.original_chiral_neighbors is not None:
            center.chiral_neighbors = tuple(stereo_update.original_chiral_neighbors)

        mol_set.update_tetrahedral_chirality(
            stereo_update.center_idx,
            ligand_replacements=stereo_update.ligand_replacements,
            stereo_mode=stereo_update.stereo_mode,
        )

    def compute_product(self):
        processed_arrows = [
            self.process_smiles_arrow(a, self.ms.atom_map_dict)
            for a in self.smiles_arrows
        ]
        processed_stereo_updates = [
            self.process_stereo_update(stereo_string, self.ms.atom_map_dict)
            for stereo_string in self.stereo_update_strings
        ]

        self._ms_prod = self.ms.make_move(processed_arrows)
        for stereo_update in processed_stereo_updates:
            self.apply_stereo_update(self._ms_prod, stereo_update)
        self._prod = self._ms_prod.can_smiles

    @property
    def prod(self):
        if self._prod is None:
            if self.check_validity():
                self.compute_product()
        return self._prod

    @property
    def ms_prod(self):
        if self._ms_prod is None:
            if self.check_validity():
                self.compute_product()
        return self._ms_prod

    def process_smiles_arrow(self, arrow_smiles, atom_map_dict):
        # text has either a form of (a, b), (a, b, "retain"), ((a, b), b),
        # ((a, b), c), ((a, b), c, "invert"), or (hv, (a, b))
        tup = self.parse_arrow_tuple(arrow_smiles)

        if not isinstance(tup, tuple):
            return ()

        if isinstance(tup[0], int):  # attack move
            if len(tup) == 2:
                return ("a", atom_map_dict[tup[0]], atom_map_dict[tup[1]])
            if len(tup) == 3 and isinstance(tup[2], str):
                return (
                    "a",
                    atom_map_dict[tup[0]],
                    atom_map_dict[tup[1]],
                    tup[2],
                )
            raise ValueError(
                "Attack move must have format (a, b) or (a, b, 'retain'|'invert'|'clear'|'unknown')"
            )

        elif isinstance(tup[0], tuple):
            # If tup[0][1] == tup[1], it's a ionization move
            if not len(tup[0]) == 2:
                raise ValueError(
                    "Move must either have a format (a, b), ((a, b), c) or (hv, (a, b))"
                )

            if len(tup) == 2 and tup[0][1] == tup[1]:
                return ("i", atom_map_dict[tup[0][0]], atom_map_dict[tup[0][1]])

            if len(tup) == 2:
                return (
                    "ba",
                    atom_map_dict[tup[0][0]],
                    atom_map_dict[tup[0][1]],
                    atom_map_dict[tup[1]],
                )
            if len(tup) == 3 and isinstance(tup[2], str):
                return (
                    "ba",
                    atom_map_dict[tup[0][0]],
                    atom_map_dict[tup[0][1]],
                    atom_map_dict[tup[1]],
                    tup[2],
                )
            raise ValueError(
                "Bond-attack move must have format ((a, b), c) or ((a, b), c, 'retain'|'invert'|'clear'|'unknown')"
            )

        elif isinstance(tup[0], str) and tup[0] == "hv":
            return ("hv", atom_map_dict[tup[1][0]], atom_map_dict[tup[1][1]])

        else:
            raise ValueError(
                "Move must either have a format (a, b), ((a, b), c) or (hv, (a, b))"
            )

    @classmethod
    def drawn_arrows_to_mechSmiles(cls, smiles, drawn_arrows):
        """
        Convert drawn arrows to MechSmiles format.
        The tuples are treated as follow:
        first the tuples can be:
        - (a,b)
        - ((a,b),c)
        - ((a,b),(c,d))
        - (a,(b,c)) : Authorized only if a is b or c, in which case a attacks the other
        """
        assert isinstance(smiles, str), "smiles must be a string"
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        assert mol is not None, "Invalid SMILES string"

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

        assert isinstance(drawn_arrows, list), "drawn_arrows must be a list of tuples"
        for arrow in drawn_arrows:
            assert isinstance(arrow, tuple), "drawn_arrows must be a list of tuples"
            assert len(arrow) == 2, "drawn_arrows must be a list of tuples of length 2"
            assert isinstance(arrow[0], tuple) or isinstance(
                arrow[0], int
            ), "elements of drawn_arrows must be either a tuple or an int"
            assert isinstance(arrow[1], tuple) or isinstance(
                arrow[1], int
            ), "elements of drawn_arrows must be either a tuple or an int"

        final_arrows = []
        last_atom_attacked = None

        for drawn_arrow in drawn_arrows:
            # (a1, a2)
            if isinstance(drawn_arrow[0], int) and isinstance(drawn_arrow[1], int):
                # Ensure that the last attacked atom is None (An attack has to initiate)
                if last_atom_attacked is None:
                    final_arrows.append((drawn_arrow[0], drawn_arrow[1]))

                else:
                    raise NotInitiatingAttackError(
                        f"The drawn arrow {drawn_arrow} for SMILES {smiles} is not valid, an attack move has to initiate"
                    )

                last_atom_attacked = drawn_arrow[1]

            # ((a1, a2), a3)
            elif isinstance(drawn_arrow[0], tuple) and isinstance(drawn_arrow[1], int):
                # The bonds can have multiple forms.
                # If a2 == a3, it's a ionization move and we can keep it as is
                if drawn_arrow[0][1] == drawn_arrow[1]:
                    final_arrows.append(drawn_arrow)
                # If a1 == a3, same but we reverse the order in the bond
                elif drawn_arrow[0][0] == drawn_arrow[1]:
                    final_arrows.append(
                        ((drawn_arrow[0][1], drawn_arrow[0][0]), drawn_arrow[1])
                    )

                # If None of the above but the last attacked is in the bond:
                elif (
                    last_atom_attacked is not None
                    and last_atom_attacked in drawn_arrow[0]
                ):
                    other_atom = (
                        drawn_arrow[0][0]
                        if last_atom_attacked == drawn_arrow[0][1]
                        else drawn_arrow[0][1]
                    )
                    final_arrows.append(
                        ((last_atom_attacked, other_atom), drawn_arrow[1])
                    )

                # If None of the above, it becomes ambiguous, we will believe the user
                else:
                    final_arrows.append(drawn_arrow)

                last_atom_attacked = drawn_arrow[1]

            # ((a1, a2), (a3, a4))
            elif isinstance(drawn_arrow[0], tuple) and isinstance(
                drawn_arrow[1], tuple
            ):
                # Find overlapping atoms
                intersection_list = list(
                    set(drawn_arrow[0]).intersection(set(drawn_arrow[1]))
                )
                assert (
                    len(intersection_list) == 1
                ), f"The drawn arrow must have exactly one common atom, arrow {drawn_arrow} for SMILES {smiles} has {len(intersection_list)} common atoms"
                common_atom = intersection_list[0]
                other_atom_1 = (
                    drawn_arrow[0][0]
                    if common_atom == drawn_arrow[0][1]
                    else drawn_arrow[0][1]
                )
                other_atom_2 = (
                    drawn_arrow[1][0]
                    if common_atom == drawn_arrow[1][1]
                    else drawn_arrow[1][1]
                )

                # We rewrite it as ((other_atom_1, common_atom), other_atom_2)
                final_arrows.append(((other_atom_1, common_atom), other_atom_2))

                last_atom_attacked = other_atom_2

            # (a, (b, c))
            elif isinstance(drawn_arrow[0], int) and isinstance(drawn_arrow[1], tuple):
                assert (
                    drawn_arrow[0] in drawn_arrow[1]
                ), f"AmbiguousError: An atom can only attack a bond if it is adjacent to it, drawn arrow {drawn_arrow} for SMILES {smiles} is not valid"

                if drawn_arrow[0] == drawn_arrow[1][0]:
                    final_arrows.append((drawn_arrow[1][0], drawn_arrow[1][1]))
                elif drawn_arrow[0] == drawn_arrow[1][1]:
                    final_arrows.append((drawn_arrow[1][1], drawn_arrow[1][0]))
                else:
                    raise AssertionError(
                        f"An atom can only attack a bond if it is adjacent to it, drawn arrow {drawn_arrow} for SMILES {smiles} is not valid"
                    )

            else:
                # Illegal (too ambiguous)
                raise ValueError(
                    f"The drawn arrow {drawn_arrow} for SMILES {smiles} is not valid, because it is ambiguous"
                )

        # Increment all indices in the moves by 1
        all_used_indices = set()

        def increment_tuple(tup):
            if isinstance(tup, int):
                all_used_indices.add(tup + 1)
                return tup + 1
            elif isinstance(tup, tuple):
                return tuple(increment_tuple(i) for i in tup)

        pre_final_arrows = [increment_tuple(arrow) for arrow in final_arrows]
        lower_int_convert = {k: (i + 1) for i, k in enumerate(all_used_indices)}

        def convert_tuple(tup, dic):
            if isinstance(tup, int):
                return dic[tup]
            elif isinstance(tup, tuple):
                return tuple(convert_tuple(i, dic) for i in tup)

        final_arrows = [
            convert_tuple(arrow, lower_int_convert) for arrow in pre_final_arrows
        ]

        print(
            f"before filtering for mapping, smiles is {Chem.MolToSmiles(mol, kekuleSmiles=True)}, arrows are {final_arrows}, all_used_indices are {all_used_indices}"
        )
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() not in all_used_indices:
                atom.SetAtomMapNum(0)
            else:
                atom.SetAtomMapNum(lower_int_convert[atom.GetAtomMapNum()])

        mech_smiles_string = f"{Chem.MolToSmiles(mol, kekuleSmiles=True)}|{';'.join([str(a) for a in final_arrows])}"
        print(f"{Fore.GREEN}MechSmiles string: {mech_smiles_string}{Fore.RESET}")
        return mech_smiles_string

    def __repr__(self):
        return f'MechSmiles("{self.value}")'


if __name__ == "__main__":
    verbose_tests = True
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
    msmi_8 = MechSmiles("[C]=O.C=C1C[OH:1].C[S+:2](C)Cl.O=C=O.[Cl-]|(1, 2)")
    msmi_8.standardize(verbose=verbose_tests)

    exit()

    # Hide/show observer
    msmi = MechSmiles(
        "CCOP(=O)(C[O-:1])OCC.NC1=C2N=C([C:2][Br:3])N(CCC3=CC=CC=C3)C2=NC=N1.[H][H].[Na+]|(1, 2);((2, 3), 3)"
    )
    msmi.show()
    msmi.hide_cond()
    print(msmi.value)
    msmi.show()
    msmi.unhide_cond()
    msmi.show()

    # Example usage
    my_move = MechSmiles("C[C:2](=[O:3])C.[NH3:1]|(1,2);((2,3), 3)")
    # my_move = MechSmiles("C[C:3](=[O:4])C.[BH3-:1][H:2]|((1,2),3);((3,4), 4)")
    # my_move = MechSmiles("C[C:2](=[O:3])C.[NH3:1]|")
    my_move.show()
