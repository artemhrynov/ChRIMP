# %% [markdown]
# # This notebook has for goal to translate the USPTO-31k to MechSMILES format, which is simply another more compact representation of equivalent information
# # Moreover, it will try to extract every MechSMILES elementary steps out of their multi-elementary-step reactions.
#
#
# The conventions go as follow:
#
# | Convention               | MechSMILES     | PMechDB     | mech-USPTO-31k                                      |
# |:------------------------:|:--------------:|:-----------:|:---------------------------------------------------:|
# | Attack                   | (a,b)          | "a=b"       | (a,b) if bond-order == 0 else (a,[a,b])             |
# | Ionization               | ((a,b), b)     | "a,b=b"     | ([a,b], b)                                          |
# | Bond-attack              | ((a,b), c)     | "b,a=b,c"   | ([a,b], c) if bond-order == 0 else ([a,b], [b,c])   |
# | Homo-cleavage            | (hv, (a,b))    | ?           | ?                                                   |
#
# Note: USPTO-31K doesn't require to explicitely map hydrogens, and gives an "implicit" index N.n for the n-th H attached to atom mapped with index N.

# %% Imports

import os
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
from typing import Tuple
import re
from rdkit import Chem
from colorama import Fore
from chrimp.world.molecule_set import (
    BondNotFoundError,
    ReusedVirtualTSException,
    MoleculeSet,
)


from chrimp.world.mechsmiles import MechSmiles
from pandarallel import pandarallel

MoleculeSet.default_treat_Br_I_2nd_period = True
MoleculeSet.default_treat_P_S_Cl_12_electrons = True

pandarallel.initialize(nb_workers=15, progress_bar=True)
# %% Load and clean the csv

df = pd.read_csv(os.path.join("data", "uspto-31k", "original", "mech-USPTO-31k.csv"))
df[["reac_smiles", "cond_smiles", "prod_smiles"]] = df["updated_reaction"].str.split(
    ">", expand=True
)

# Remove trivial duplicates
total_len = len(df)
df_no_dupl = df.drop_duplicates(
    subset=["reac_smiles", "prod_smiles", "mechanistic_label"]
)
len_no_dupl = len(df_no_dupl)

# Remove duplicates (trivial)
df_rxn = df_no_dupl[
    df_no_dupl.apply(lambda x: x["reac_smiles"] != x["prod_smiles"], axis=1)
]
remaining_len = len(df_rxn)

print(f"Total len: {total_len:_}")
print(f"Removing {total_len-len_no_dupl:_} duplicates")
print(f"Removing {len_no_dupl-remaining_len:_} trivial")
print(f"Remaining: {remaining_len:_} ({remaining_len/total_len:.2%})")

# %% Define the verification and extraction of every elementary step:


def process_uspto_31k_arrow(arrow: Tuple, verbose=False):
    if isinstance(arrow[0], (int, float)):
        # Attack (bond-order == 0)
        if isinstance(arrow[1], (int, float)):
            if verbose:
                print(str(arrow) + "(a, bo=0)")
            return str(arrow)
        # Attack (bond-order > 0)
        elif isinstance(arrow[1], list):
            assert len(arrow[1]) == 2, f"Type of arrow {arrow} not recognized"
            assert arrow[0] == arrow[1][0], f"Type of arrow {arrow} not recognized"
            if verbose:
                print(f"({arrow[0]}, {arrow[1][1]}) (a, bo>0)")
            return f"({arrow[0]}, {arrow[1][1]})"
        else:
            raise ValueError(f"Type of arrow {arrow} not recognized")

    elif isinstance(arrow[0], list):
        assert len(arrow[0]) == 2, f"Type of arrow {arrow} not recognized"

        if isinstance(arrow[1], (int, float)):
            # Ionization
            if arrow[0][1] == arrow[1]:
                if verbose:
                    print(f"(({arrow[0][0]},{arrow[1]}),{arrow[1]}) (i)")
                return f"(({arrow[0][0]},{arrow[1]}),{arrow[1]})"

            # Bond-attack (bond-order == 0)
            else:
                if verbose:
                    print(f"(({arrow[0][0]}, {arrow[0][1]}),{arrow[1]}) (ba bo=0)")
                return f"(({arrow[0][0]}, {arrow[0][1]}),{arrow[1]})"

        # Bond-attack (bond-order > 0)
        elif isinstance(arrow[1], list):
            assert len(arrow[0]) == 2, f"Type of arrow {arrow} not recognized"
            assert len(arrow[1]) == 2, f"Type of arrow {arrow} not recognized"
            assert arrow[0][1] == arrow[1][0], f"Type of arrow {arrow} not recognized"
            if verbose:
                print(f"(({arrow[0][0]}, {arrow[0][1]}), {arrow[1][1]}) (ba bo>0)")
            return f"(({arrow[0][0]}, {arrow[0][1]}), {arrow[1][1]})"

        else:
            raise ValueError(f"Type of arrow {arrow} not recognized")

    else:
        raise NotImplementedError(f"Type of arrow {arrow} not recognized")


def get_indices_from_tuple(x):
    if isinstance(x, (int, float)):
        return {x}

    elif isinstance(x, tuple):
        return set().union(*[get_indices_from_tuple(x_i) for x_i in x])


def rewrite_new_tuple(x, replacement_dict):
    if isinstance(x, str):
        return rewrite_new_tuple(literal_eval(x), replacement_dict)
    elif isinstance(x, (int, float)):
        return replacement_dict.get(x, x)
    elif isinstance(x, tuple):
        return tuple([rewrite_new_tuple(x_i, replacement_dict) for x_i in x])
    else:
        raise ValueError(f"Cannot recognize the type of {x}")


def get_n_indices_not_in_given_set(n, given_set):
    MAX_ITER = 10**9
    i = 1
    num_found = 0
    list_indices = []

    assert (
        n < MAX_ITER
    ), f"n is too big, surpasses {MAX_ITER = }, you should probably not use that function for such big n"
    for _ in range(MAX_ITER):
        if i not in given_set:
            list_indices.append(i)
            num_found += 1
        i += 1
        if num_found >= n:
            break

    return list_indices


def unmap_given_indices(smiles, given_indices):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    for a in mol.GetAtoms():
        if a.GetAtomMapNum() in given_indices:
            a.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol)


def unmap_linked_hs(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    for a in mol.GetAtoms():
        if a.GetSymbol() == "H" and a.GetAtomMapNum() > 0:
            b: Chem.Bond = a.GetBonds()[0]
            if not all(
                [b.GetOtherAtom(a).GetSymbol() == "H"]
            ):  # At least linked to one heavy atom
                a.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol)


def expand_one_h(smiles: str, x: int, y: int, first_run: bool = True) -> str:
    """
    Add one explicit hydrogen to the atom whose AtomMapNum==x
    and assign that hydrogen AtomMapNum==y.
    Returns a mapped SMILES with that explicit [H:y] attached.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Bad input SMILES")

    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
    )

    # if first_run:
    mol = Chem.AddHs(mol)

    atom_of_interest = None
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == x:
            atom_of_interest = a
            break

    if atom_of_interest is None:
        raise ValueError(f"Couldn't find atom mapped with {x} in {smiles}")

    bond_found = False
    for bond in atom_of_interest.GetBonds():
        other_atom = bond.GetOtherAtom(atom_of_interest)
        if other_atom.GetSymbol() == "H":
            other_atom.SetAtomMapNum(y)
            bond_found = True
            break

    if not bond_found:
        raise ValueError(f"Could not find an H atom attached to atom {x} in {smiles}")

    return Chem.MolToSmiles(mol, canonical=True)


def get_all_floats_in_tuple_list(tup_list):
    # Max depth of these tuples is 2
    all_floats = []

    for tup in tup_list:
        all_floats.extend(get_all_floats_in_tuple(tup))
    return list(set(all_floats))


def get_all_floats_in_tuple(tup):
    all_floats = []

    if isinstance(tup, str):
        tup = literal_eval(tup)

    for elem in tup:
        if isinstance(elem, float):
            all_floats.append(elem)
        elif isinstance(elem, (tuple, list)):
            all_floats.extend(get_all_floats_in_tuple(elem))

    return all_floats


def verify_extract_uspto_31k(
    mapped_rxn_reac, mapped_rxn_prod, arrow_sequence, verbose=False
):
    arrows_mechsmiles = [process_uspto_31k_arrow(a) for a in arrow_sequence]
    # print(arrows_mechsmiles)

    all_indices_reac = set([int(x) for x in re.findall(r":(\d+)]", mapped_rxn_reac)])
    all_indices_arrows = set().union(
        *[get_indices_from_tuple(literal_eval(a)) for a in arrows_mechsmiles]
    )

    indices_implicit_hs = all_indices_arrows - all_indices_reac

    # print(all_indices_reac)
    # print(all_indices_arrows)
    # print(indices_implicit_hs)

    # Get indices for these Hs
    new_indices_implicit_hs = get_n_indices_not_in_given_set(
        len(indices_implicit_hs), all_indices_reac
    )

    # print(new_indices_implicit_hs)

    dict_of_replacements = dict(zip(indices_implicit_hs, new_indices_implicit_hs))

    first_run = True
    # print(f"Old reac SMILES:\n{mapped_rxn_reac}")
    # print(f"New reac SMILES:\n{mapped_rxn_reac}")

    # print(arrows_mechsmiles)

    arrow_flows = []
    stable_ints = []

    while len(arrows_mechsmiles) > 0:
        j = 0
        while j < len(arrows_mechsmiles):
            j += 1
            picked_arrows = arrows_mechsmiles[:j]

            # print(f"Before get_all_floats")
            # Check if implicit hydrogens are in the picked_arrows
            all_floats = get_all_floats_in_tuple_list(picked_arrows)
            # print(f"{all_floats = }")

            if verbose:
                print(f"Trying {picked_arrows} on {mapped_rxn_reac}")

            mapped_rxn_reac = unmap_given_indices(
                mapped_rxn_reac, new_indices_implicit_hs
            )
            for old_idx in all_floats:
                if verbose:
                    print(f"{Fore.GREEN}{all_floats}{Fore.RESET}")
                old_index_heavy, sub_index = (
                    int(str(old_idx).split(".")[0]),
                    int(str(old_idx).split(".")[1]),
                )
                if sub_index != 1:
                    print(
                        f"Sub-index not 1 in:\n{mapped_rxn_reac = }\n{picked_arrows = }"
                    )
                mapped_rxn_reac = expand_one_h(
                    mapped_rxn_reac,
                    old_index_heavy,
                    dict_of_replacements[old_idx],
                    first_run=first_run,
                )
                first_run = False

            picked_arrows = [
                str(rewrite_new_tuple(a, dict_of_replacements)) for a in picked_arrows
            ]
            arrow_flow = ";".join(picked_arrows)
            try:
                # Check if we trigger the ReusedVirtualTSException
                msmi = MechSmiles(f"{mapped_rxn_reac}|{arrow_flow}")
                ms_prod_found = msmi.ms_prod

                if verbose:
                    print(f"found {arrow_flow = }")
                # If not, let's add that move
                stable_ints.append(mapped_rxn_reac)
                arrow_flows.append(arrow_flow)
                arrows_mechsmiles = arrows_mechsmiles[j:]

                # And update next stable intermediate (mapped, except for implicit hydrogens)
                mapped_rxn_reac = unmap_given_indices(
                    ms_prod_found.mapped_smiles, new_indices_implicit_hs
                )

                break

            except ReusedVirtualTSException:
                pass

            except Chem.AtomValenceException:
                pass

            except BondNotFoundError:
                pass

        # if 0 < len(arrows_mechsmiles) < j :
        #    raise ValueError("Couldn't find a legal MechSMILES path")

    return stable_ints, arrow_flows


# %% Few tricky reactions
# Do we want to try and fix this ones too? Timeout was set to 10s
# """
# Timeout with args = ('[CH3:3][Si:4]([CH3:5])([CH3:6])[CH2:7][CH2:8][O:9][CH2:10][n:11]1[cH:12][n:13][c
#:14](-[c:15]2[cH:16][cH:17][c:18]([N+:19](=[O:20])[OH:21])[cH:22][cH:23]2)[cH:24]1.C[N:101](C)[CH:1]=[
# O:2].[O:301]=[P:302](Cl)(Cl)[Cl:303].[OH2:304]', '[CH:1](=[O:2])[c:12]1[n:11]([CH2:10][O:9][CH2:8][CH2
#:7][Si:4]([CH3:3])([CH3:5])[CH3:6])[cH:24][c:14](-[c:15]2[cH:16][cH:17][c:18]([N+:19](=[O:20])[OH:21])
# [cH:22][cH:23]2)[n:13]1', '[(101, [101, 1]), ([1, 2], 302), ([302, 301], 301), (301, [301, 302]), ([30
# 2, 303], 303), (303, 1), ([1, 101], 101), (101, [101, 1]), ([1, 2], [2, 302]), ([302, 301], 301), ([11
# , 12], 1), ([1, 101], 101), (301, 12.1), ([12.1, 12], [12, 11]), (101, [101, 1]), ([1, 303], 303), (30
# 4, 1), ([1, 101], 101), (303, 304.1), ([304.1, 304], [304, 1]), ([1, 101], 101), (101, 304.1), ([304.1
# , 304], 304)]')
# """

df_rxn.iloc[164]
df_rxn.iloc[164]["updated_reaction"]
eval(df_rxn.iloc[164]["mechanistic_label"])

# Inconstistant number of atoms ("[C]=O" transforming into "C=O")
a, b = verify_extract_uspto_31k(
    df_rxn.iloc[164]["updated_reaction"].split(">")[0],
    df_rxn.iloc[164]["updated_reaction"].split(">")[-1],
    literal_eval(df_rxn.iloc[164]["mechanistic_label"]),
    verbose=True,
)

print("\n".join([MechSmiles(f"{a_i}|{b_i}").standard_value for a_i, b_i in zip(a, b)]))
print("\n".join([MechSmiles(f"{a_i}|{b_i}").value for a_i, b_i in zip(a, b)]))


df_rxn[
    df_rxn["reac_smiles"]
    == "[OH:101][CH:1]([CH2:2][CH3:3])[c:4]1[n:5][n:6]2[cH:7][cH:8][cH:9][c:10]2[c:11](=[O:12])[n:13]1[CH2:14][c:15]1[cH:16][cH:17][cH:18][cH:19][cH:20]1.[CH3:21][C:22]([CH3:23])([CH3:24])[O:25][C:26](=[O:27])[NH:28][CH2:29][CH2:30][CH2:31][NH2:32].[O:301]=[S:302]([Cl:303])(=O)c1ccc(cc1)C"
]["mechanistic_label"].values[0]

# With this one, several Hs are ending up having the same index ?!
a, b = verify_extract_uspto_31k(
    "[OH:101][CH:1]([CH2:2][CH3:3])[c:4]1[n:5][n:6]2[cH:7][cH:8][cH:9][c:10]2[c:11](=[O:12])[n:13]1[CH2:14][c:15]1[cH:16][cH:17][cH:18][cH:19][cH:20]1.[CH3:21][C:22]([CH3:23])([CH3:24])[O:25][C:26](=[O:27])[NH:28][CH2:29][CH2:30][CH2:31][NH2:32].[O:301]=[S:302]([Cl:303])(=O)c1ccc(cc1)C",
    "[CH:1]([CH2:2][CH3:3])([c:4]1[n:5][n:6]2[cH:7][cH:8][cH:9][c:10]2[c:11](=[O:12])[n:13]1[CH2:14][c:15]1[cH:16][cH:17][cH:18][cH:19][cH:20]1)[NH:32][CH2:31][CH2:30][CH2:29][NH:28][C:26]([O:25][C:22]([CH3:21])([CH3:23])[CH3:24])=[O:27]",
    [
        (101, 302),
        ([302, 301], 301),
        (301, [301, 302]),
        ([302, 303], 303),
        (303, 101.1),
        ([101.1, 101], 101),
        (32, 1),
        ([1, 101], 101),
        (101, 32.1),
        ([32.1, 32], 32),
    ],
    verbose=True,
)
print("\n".join([MechSmiles(f"{a_i}|{b_i}").standard_value for a_i, b_i in zip(a, b)]))
print("\n".join([MechSmiles(f"{a_i}|{b_i}").value for a_i, b_i in zip(a, b)]))


# With this one, several Hs are ending up having the same index ?!
a, b = verify_extract_uspto_31k(
    "[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([C:7](=[O:8])[CH2:9][c:10]2[c:11]([Cl:12])[cH:13][n+:14]([OH:15])[cH:16][c:17]2[Cl:18])[c:19]2[c:20]1[O:21][C:22]1([CH2:23][CH2:24][CH2:25][CH2:26]1)[O:27]2.[BH4-:301].[CH3][OH1:302]",
    "[CH3:1][O:2][c:3]1[cH:4][cH:5][c:6]([CH:7]([OH:8])[CH2:9][c:10]2[c:11]([Cl:12])[cH:13][n+:14]([OH:15])[cH:16][c:17]2[Cl:18])[c:19]2[c:20]1[O:21][C:22]1([CH2:23][CH2:24][CH2:25][CH2:26]1)[O:27]2",
    [([301, 301.1], 7), ([7, 8], 8), (8, 302.1), ([302.1, 302], 302)],
    verbose=True,
)
print(a)
print(b)

# Still a bug with this one, we would need the initial atom map on lone Hs to stick even when they are not lone anymore
a, b = verify_extract_uspto_31k(
    "[OH:101][C:4]([C@H:2]([C@H:1]([OH:6])[CH2:7][CH2:8][CH2:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1)[CH3:3])=[O:5].[H+:301]",
    "[C@@H:1]1([CH2:7][CH2:8][CH2:9][CH2:10][c:11]2[cH:12][cH:13][cH:14][cH:15][cH:16]2)[C@H:2]([CH3:3])[C:4](=[O:5])[O:6]1",
    [
        (5, 301),
        (6, 4),
        ([4, 5], 5),
        (101, 6.1),
        ([6.1, 6], 6),
        (5, [5, 4]),
        ([4, 101], 101),
        ([301, 5], 5),
    ],
    verbose=True,
)
print(a)
print(b)

# Still a bug with this one, we would need the initial atom map on lone Hs to stick even when they are not lone anymore
a, b = verify_extract_uspto_31k(
    "CO[N:101](C)[C:1](=[O:2])[c:3]1[cH:4][s:5][c:6]([NH:7][C:8](=[O:9])[O:10][C:11]([CH3:12])([CH3:13])[CH3:14])[n:15]1.[Mg+:102][CH2:16][CH3:17].[H+:301].[H+:302]",
    "[C:1](=[O:2])([c:3]1[cH:4][s:5][c:6]([NH:7][C:8](=[O:9])[O:10][C:11]([CH3:12])([CH3:13])[CH3:14])[n:15]1)[CH2:16][CH3:17]",
    [
        ([102, 16], 1),
        ([1, 2], 102),
        ([102, 2], 301),
        (101, 302),
        ([301, 2], [2, 1]),
        ([1, 101], 101),
    ],
    verbose=True,
)
print(a)
print(b)


a, b = verify_extract_uspto_31k(
    "[CH3:5][C:6](=[O:7])[O:8][CH:9]([CH2:10][O:11][CH2:12][CH2:13][OH:14])[c:15]1[cH:16][cH:17][cH:18][c:19]([Cl:20])[cH:21]1.[Cl:101][S:1]([CH3:2])(=[O:3])=[O:4].[H-:301].[Na+]",
    "[S:1]([CH3:2])(=[O:3])(=[O:4])[O:14][CH2:13][CH2:12][O:11][CH2:10][CH:9]([O:8][C:6]([CH3:5])=[O:7])[c:15]1[cH:16][cH:17][cH:18][c:19]([Cl:20])[cH:21]1",
    [(301, 14.1), ([14.1, 14], 14), (14, 1), ([1, 3], 3), (3, [3, 1]), ([1, 101], 101)],
)
print(a)
print(b)

# Needs some halogens to be treated like 2nd period
a, b = verify_extract_uspto_31k(
    "[OH:101][CH2:2][CH2:3][C@@:4]1([C:5](=[O:6])[O:7][C:8]([CH3:9])([CH3:10])[CH3:11])[CH2:12][N:13]([C@H:14]([CH3:15])[c:16]2[cH:17][cH:18][cH:19][cH:20][cH:21]2)[C:22](=[O:23])[CH:24]1[F:25].Br[C:102](Br)(Br)[Br:1].c1ccccc1[P:301](c2ccccc2)c3ccccc3",
    "[Br:1][CH2:2][CH2:3][C@@:4]1([C:5](=[O:6])[O:7][C:8]([CH3:9])([CH3:10])[CH3:11])[CH2:12][N:13]([C@H:14]([CH3:15])[c:16]2[cH:17][cH:18][cH:19][cH:20][cH:21]2)[C:22](=[O:23])[CH:24]1[F:25]",
    [
        (301, 1),
        ([1, 102], 102),
        (102, 101.1),
        ([101.1, 101], 101),
        (101, 301),
        ([301, 1], 1),
        (1, 2),
        ([2, 101], [101, 301]),
    ],
)
print(a)
print(b)

df_rxn[
    df_rxn["reac_smiles"]
    == "[Br:101][CH2:1][c:2]1[n:3][c:4]2[c:5]([NH2:6])[n:7][cH:8][n:9][c:10]2[n:11]1[CH2:12][CH2:13][c:14]1[cH:15][cH:16][cH:17][cH:18][cH:19]1.[CH3:20][CH2:21][O:22][P:23](=[O:24])([CH2:25][OH:26])[O:27][CH2:28][CH3:29].[H-:301].[Na+]"
]["prod_smiles"].values[0]

# H2 disappearing
a, b = verify_extract_uspto_31k(
    "[Br:101][CH2:1][c:2]1[n:3][c:4]2[c:5]([NH2:6])[n:7][cH:8][n:9][c:10]2[n:11]1[CH2:12][CH2:13][c:14]1[cH:15][cH:16][cH:17][cH:18][cH:19]1.[CH3:20][CH2:21][O:22][P:23](=[O:24])([CH2:25][OH:26])[O:27][CH2:28][CH3:29].[H-:301].[Na+]",
    "[CH2:1]([c:2]1[n:3][c:4]2[c:5]([NH2:6])[n:7][cH:8][n:9][c:10]2[n:11]1[CH2:12][CH2:13][c:14]1[cH:15][cH:16][cH:17][cH:18][cH:19]1)[O:26][CH2:25][P:23]([O:22][CH2:21][CH3:20])(=[O:24])[O:27][CH2:28][CH3:29]",
    [(301, 26.1), ([26.1, 26], 26), (26, 1), ([1, 101], 101)],
)
print(a)
print(b)


# %% Actually run the verification and extraction
all_stable_lists, all_arrow_flows = [], []

from concurrent.futures import ProcessPoolExecutor, as_completed
from ast import literal_eval
import signal

MAX_TIME = 10  # seconds per task


class TaskTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TaskTimeout()


def _worker(args):
    # This code runs in a separate process → we can use SIGALRM safely on Unix.
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MAX_TIME)
    try:
        mapped_reac, mapped_prod, mechanistic_label, row_number = args
        arrows = (
            literal_eval(mechanistic_label)
            if isinstance(mechanistic_label, str)
            else mechanistic_label
        )
        stable_ints, arrow_flows = verify_extract_uspto_31k(
            mapped_reac, mapped_prod, arrows
        )
        return stable_ints, arrow_flows, mapped_reac, mapped_prod, row_number
    except TaskTimeout:
        print(f"Timeout with {args = }")
        return None  # timed out → skip
    except Exception as e:
        print(f"Exception {e} with {args}")
        return None  # other failures → skip
    finally:
        signal.alarm(0)  # clear alarm


df_rxn["row_number"] = range(len(df_rxn))

# Prep rows once (faster than iterrows)
rows = list(
    df_rxn[
        ["reac_smiles", "prod_smiles", "mechanistic_label", "row_number"]
    ].itertuples(index=False, name=None)
)

all_stable_lists, all_arrow_flows, all_mapped_reac, all_mapped_prod, all_row_indices = (
    [],
    [],
    [],
    [],
    [],
)

with ProcessPoolExecutor(max_workers=14) as ex:
    futures = [ex.submit(_worker, row) for row in rows]
    for fut in tqdm(as_completed(futures), total=len(futures)):
        res = fut.result()  # no timeout needed here; worker enforces it
        if res is None:
            continue
        stable_ints, arrow_flows, mapped_reac, mapped_prod, row_number = res
        all_stable_lists.append(stable_ints)
        all_arrow_flows.append(arrow_flows)
        all_mapped_reac.append(mapped_reac)
        all_mapped_prod.append(mapped_prod)
        all_row_indices.append(row_number)


msmi = MechSmiles(f"{all_stable_lists[3][1]}|{all_arrow_flows[3][1]}")

# %%
data = []
for idx, (r_list, a_list, m_r, m_p, row_idx) in enumerate(
    zip(
        all_stable_lists,
        all_arrow_flows,
        all_mapped_reac,
        all_mapped_prod,
        all_row_indices,
    )
):
    for sub_idx, (r, a) in enumerate(zip(r_list, a_list)):
        data.append(
            [
                f"{r}|{a}",
                idx,
                len(r_list),
                len(r_list) - sub_idx - 1,
                m_r,
                m_p,
                f"rxn_{row_idx}",
            ]
        )

df = pd.DataFrame(
    data,
    columns=[
        "mech_smi",
        "single_step_idx",
        "single_step_length",
        "step_idx_retro",
        "single_step_original_reac",
        "single_step_original_prod",
        "source",
    ],
)
df.to_csv(
    os.path.join("data", "uspto-31k", "first_mechsmiles_extraction.csv"), index=False
)


# %%
import seaborn as sns
import matplotlib.pyplot as plt

lengths = df.groupby("single_step_idx").size().reset_index(name="single_step_length")
avg_len = lengths["single_step_length"].mean()

sns.countplot(data=lengths, x="single_step_length")
plt.xlabel("Single step length")
plt.ylabel("Number of single steps")
plt.title(
    f"Distribution of {len(lengths)} {len(lengths)/len(df_rxn):.2%} single steps, length mean = {avg_len:.4f}"
)
plt.show(block=False)

# %%
df = pd.read_csv(os.path.join("data", "uspto-31k", "first_mechsmiles_extraction.csv"))

df = df.rename(columns={"mech_smi": "mech_smi_raw"})


def safe_msmi_standardize(msmi_str):
    try:
        return MechSmiles(msmi_str).standard_value, True
    except:  # noqa: E722 (Do not use bare except)
        return msmi_str, False


df["mech_smi_tuple"] = df["mech_smi_raw"].parallel_apply(
    lambda x: safe_msmi_standardize(x)
)

df[["mech_smi", "could_canonicalize"]] = pd.DataFrame(
    df["mech_smi_tuple"].tolist(), index=df.index
)
df = df.drop(columns=["mech_smi_tuple"])
df.to_csv(
    os.path.join("data", "uspto-31k", "first_mechsmiles_extraction_intermediate.csv"),
    index=False,
)


# %%
df = pd.read_csv(
    os.path.join("data", "uspto-31k", "first_mechsmiles_extraction_intermediate.csv")
)

df["could_canonicalize"].value_counts()
df = df.drop(columns=["could_canonicalize"])

df["reac_elem_step"] = df["mech_smi"].parallel_apply(
    lambda x: MechSmiles(x).ms.can_smiles
)


# %%
def safe_prod(msmi_str, msmi_raw):
    msmi = MechSmiles(msmi_str)

    try:
        return msmi.ms_prod.can_smiles
    except Exception as e:
        print(f"Problem {e} with:\n{msmi_str = }\n {msmi_raw = }")
        return ""


df["prod_elem_step"] = df.parallel_apply(
    lambda x: safe_prod(x["mech_smi"], x["mech_smi_raw"]), axis=1
)

# %%
df.columns

df_ds1 = df[["mech_smi", "single_step_idx", "step_idx_retro", "source"]]
df_ds1 = df_ds1.rename(columns={"single_step_idx": "rxn_idx"})
df_ds1["source_db"] = "mech_uspto_31k"

df_ds1.columns


df_ds1.to_csv(os.path.join("data", "preprocessed", "mech_uspto_31k.csv"), index=False)
