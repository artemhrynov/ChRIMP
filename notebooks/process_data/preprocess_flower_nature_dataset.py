# %% Imports

import pandas as pd
import os
from rdkit import Chem
import numpy as np

from chrimp.world.mechsmiles import MechSmiles
from chrimp.world.molecule_set import ReusedVirtualTSException

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import signal

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=15, progress_bar=True)

MAX_TIME = 10  # seconds per task
MAX_WORKERS = 20

# %% Load data

# split = "train"
# split = "val"
split = "test"

df_raw = pd.read_csv(
    os.path.join(
        "data",
        "flower_nature",
        "original",
        "data",
        "flower_new_dataset",
        f"{split}.txt",
    ),
    header=None,
    sep="|",
    names=["elem_rxn", "rxn_idx"],
)


# %%
df = df_raw.copy()
len_df_all = len(df)


# Not sure yet what the letters rxn_idx mean
df = df[df["rxn_idx"].apply(lambda x: isinstance(x, int) or x.isdigit())]
df["rxn_idx"] = [int(x) for x in df["rxn_idx"]]
len_df_digit = len(df)


# Remove trivial elementary steps (effectively acting as stop tokens for diffusion models probably)
df = df[df["elem_rxn"].apply(lambda x: x.split(">")[0] != x.split(">")[-1])]
len_df_digit_non_trivial = len(df)


print(f"Len of df                    : {len_df_all:_}")
print(
    f"Len of df with digits indices: {len_df_digit:_} ({len_df_digit/len_df_all:.2%} of previous)"
)
print(
    f"And non-trivial              : {len_df_digit_non_trivial:_} ({len_df_digit_non_trivial/len_df_digit:.2%} of previous)"
)

# %%


def check_cond(rxn, keep_cond=True):
    reac, prod = rxn.split(">")[::2]

    set_reac = set(reac.split("."))
    set_prod = set(prod.split("."))
    set_cond = set_reac.intersection(set_prod)

    return f"{'.'.join(set_reac-set_cond)}>{'.'.join(set_cond if keep_cond else [])}>{'.'.join(set_prod-set_cond)}"


def mapped_elem_to_mechsmiles(mapped_rxn, verbose=False):
    reac, cond, prod = check_cond(mapped_rxn, keep_cond=True).split(">")

    reac_mol = Chem.MolFromSmiles(reac, sanitize=False)
    prod_mol = Chem.MolFromSmiles(prod, sanitize=False)

    num_atoms = reac_mol.GetNumAtoms()

    if num_atoms != prod_mol.GetNumAtoms():
        raise ValueError("Not same numbers of atom on each side")

    if num_atoms == 0:
        raise NotImplementedError("Do something about it as it is the [eos] of flower")

    charge_reac = np.zeros(num_atoms)
    charge_prod = np.zeros(num_atoms)

    map_dict = dict()

    bonds_reac = np.zeros((num_atoms, num_atoms))
    bonds_prod = np.zeros((num_atoms, num_atoms))

    for i, atom in enumerate(reac_mol.GetAtoms()):
        a: Chem.Atom = atom
        map_dict[a.GetAtomMapNum()] = i

    for i, atom in enumerate(reac_mol.GetAtoms()):
        a: Chem.Atom = atom
        charge_reac[map_dict[a.GetAtomMapNum()]] = a.GetFormalCharge()
        for bond in a.GetBonds():
            b: Chem.Bond = bond
            deg = float(b.GetBondTypeAsDouble())
            if deg != int(deg):
                return "", "Aromatic bond detected"
                raise ValueError(f"Aromatic bond detected with {mapped_rxn}")
            bonds_reac[
                map_dict[a.GetAtomMapNum()], map_dict[b.GetOtherAtom(a).GetAtomMapNum()]
            ] = int(deg)

    for i, atom in enumerate(prod_mol.GetAtoms()):
        a: Chem.Atom = atom
        charge_prod[map_dict[a.GetAtomMapNum()]] = a.GetFormalCharge()
        for bond in a.GetBonds():
            b: Chem.Bond = bond
            deg = float(b.GetBondTypeAsDouble())
            if deg != int(deg):
                return "", "Aromatic bond detected"
                raise ValueError(f"Aromatic bond detected with {mapped_rxn}")
            bonds_prod[
                map_dict[a.GetAtomMapNum()], map_dict[b.GetOtherAtom(a).GetAtomMapNum()]
            ] = int(deg)

    charge_diff = charge_prod - charge_reac
    bonds_diff = bonds_prod - bonds_reac

    rev_map_dict = {v: k for k, v in map_dict.items()}

    import networkx as nx

    adj_matrix = bonds_diff * bonds_diff
    G = nx.from_numpy_array(adj_matrix)

    for i in range(len(bonds_diff)):
        for j in range(len(bonds_diff[0])):
            if G.has_edge(i, j):
                G[i][j]["bond_diff"] = bonds_diff[i, j]

    for k in range(len(charge_diff)):
        G.nodes[k]["charge_diff"] = charge_diff[k]
        G.nodes[k]["map_idx"] = rev_map_dict[k]

    # Count nodes with charge_diff of -1 and +1
    nodes_plus_one = [
        node for node, data in G.nodes(data=True) if data["charge_diff"] == +1
    ]
    nodes_minus_one = [
        node for node, data in G.nodes(data=True) if data["charge_diff"] == -1
    ]

    path_found = False
    # If there's exactly one node of each, find the shortest path
    if len(nodes_minus_one) == 1 and len(nodes_plus_one) == 1:
        source_node = nodes_plus_one[0]
        target_node = nodes_minus_one[0]

        try:
            # Find shortest path
            shortest_path = nx.shortest_path(G, source=source_node, target=target_node)

            # Collect node features (map_idx) along the path
            node_features = []
            for node in shortest_path:
                if "map_idx" in G.nodes[node]:
                    node_features.append(G.nodes[node]["map_idx"])
                else:
                    node_features.append(None)  # or handle missing feature as needed

            # Collect edge features (bond_diff) along the path
            edge_features = []
            for i in range(len(shortest_path) - 1):
                current_node = shortest_path[i]
                next_node = shortest_path[i + 1]
                edge_features.append(G[current_node][next_node]["bond_diff"])

            path_found = True

        except nx.NetworkXNoPath:
            path_found = False

            return "", "No path found"

    elif len(nodes_minus_one) == 0 and len(nodes_plus_one) == 0:
        # Loop case: find shortest cycle from node 0 starting with a -1 edge
        # e.g. in a Diels-Alder, the arrows are forming a loop
        try:
            # Find a cycle containing a node on the cycle
            cycle_edges = nx.find_cycle(G, source=next(iter(G.edges()))[0])
            # cycle_edges = nx.find_cycle(G, source=0)

            # Convert edge list to node path
            cycle_nodes = [cycle_edges[0][0]]  # Start with first node
            for edge in cycle_edges:
                cycle_nodes.append(edge[1])

            # Collect node features along the cycle
            node_features = []
            for node in cycle_nodes:
                if "map_idx" in G.nodes[node]:
                    node_features.append(G.nodes[node]["map_idx"])
                else:
                    node_features.append(None)

            # Collect edge features along the cycle
            edge_features = []
            for i in range(len(cycle_nodes) - 1):
                current_node = cycle_nodes[i]
                next_node = cycle_nodes[i + 1]
                edge_features.append(G[current_node][next_node]["bond_diff"])

            if edge_features[0] != -1:
                # Wrong direction, we reverse everything
                cycle_nodes = cycle_nodes[::-1]
                node_features = node_features[::-1]
                edge_features = edge_features[::-1]

            shortest_path = cycle_nodes

            path_found = True

        except nx.NetworkXNoCycle:
            path_found = False
            if verbose:
                print("No cycle found containing node 0")
            return "", "No cycle found"

    if path_found:
        primitive_arrows = []

        for i, ef in enumerate(edge_features):
            primitive_arrow = (
                "i" if ef == -1 else "a",
                node_features[i],
                node_features[i + 1],
            )
            primitive_arrows.append(primitive_arrow)

        arrows = []
        while len(primitive_arrows) > 0:
            # Detect bond-attacks as i followed by a on the same atom
            if (
                len(primitive_arrows) > 1
                and primitive_arrows[0][0] == "i"
                and primitive_arrows[1][0] == "a"
                and primitive_arrows[0][2] == primitive_arrows[1][1]
            ):
                pa1, pa2 = primitive_arrows[:2]
                arrows.append(f"(({pa1[1]}, {pa1[2]}), {pa2[2]})")
                primitive_arrows = primitive_arrows[2:]

            else:
                pa = primitive_arrows[0]

                if pa[0] == "a":
                    arrows.append(f"({pa[1]}, {pa[2]})")
                elif pa[0] == "i":
                    arrows.append(f"(({pa[1]}, {pa[2]}), {pa[2]})")
                else:
                    ValueError("Case not recognized")
                primitive_arrows = primitive_arrows[1:]

        mechsmiles_arrows = ";".join(arrows)
        if verbose:
            print(f"Shortest path:             {shortest_path}")
            print(f"Shortest path: (map_idx):  {node_features}")
            print(f"Edge features (bond_diff): {edge_features}")
            print(f"MechSMILES arrows        : {mechsmiles_arrows}")

        return f"{reac}{'.'+cond if cond != '' else ''}|{mechsmiles_arrows}", ""

    else:
        if verbose:
            print(
                "Condition not met: need exactly one node with charge_diff = -1 and one with charge_diff = +1 or zero of each"
            )
        # print(f"{check_cond(mapped_rxn, keep_cond=True) = }")
        # breakpoint()
        return "", f"Found {len(nodes_plus_one) = } and {len(nodes_plus_one) = }"


# %%
class TaskTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TaskTimeout()


def _worker(args):
    # This code runs in a separate process → we can use SIGALRM safely on Unix.
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MAX_TIME)
    try:
        mech_smi, error = mapped_elem_to_mechsmiles(args[0])
        return mech_smi, error
    except TaskTimeout:
        print(f"Timeout with {args = }")
        return "", f"Timeout {MAX_TIME}s"  # timed out → skip
    except Exception as e:
        print(f"Exception {e} with {args}")
        return "", str(e)  # other failures → skip
    finally:
        signal.alarm(0)  # clear alarm


# Prep rows once (faster than iterrows)
rows = list(df[["elem_rxn"]].itertuples(index=False, name=None))

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
    print("Inside ProcessPoolExecutor")
    results = list(tqdm(ex.map(_worker, rows), total=len(rows)))

all_mech_smiles = [res[0] if res else "" for res in results]
all_errors = [res[1] if res else "" for res in results]


# %%

df["mech_smi"] = all_mech_smiles
df["error_found"] = all_errors
df["error_found"].value_counts(normalize=True)
df["error_found"].value_counts()


df.to_csv(
    os.path.join("data", "flower_nature", "flower_new", f"{split}_intermediate.csv"),
    index=None,
)


# %% Analyze error type
# df = pd.read_csv(os.path.join("data", "flower_nature", "flower_new", f"{split}_intermediate.csv"))
#
# df_nocycle = df[df['error_found']=='No cycle found']
# len(df_nocycle)
# df_nocycle['min_elem_rxn'] = df_nocycle['elem_rxn'].apply(lambda x: check_cond(x))
# df_nocycle['min_elem_rxn'].values[-1:]
#
#
# df_zero_source_zero_sink = df[df['error_found']=='Found len(nodes_plus_one) = 0 and len(nodes_plus_one) = 0']
# df_zero_source_zero_sink['min_elem_rxn'] = df_zero_source_zero_sink['elem_rxn'].apply(lambda x: check_cond(x))
# df_zero_source_zero_sink['min_elem_rxn'].values[:]


# %% Verify the MechSMILES

df = pd.read_csv(
    os.path.join("data", "flower_nature", "flower_new", f"{split}_intermediate.csv")
)


# %%
def check_same_prod(rxn, msmi_string):
    if not pd.isna(msmi_string):
        try:
            mol = Chem.MolFromSmiles(rxn.split(">")[-1])
            for a in mol.GetAtoms():
                a.SetAtomMapNum(0)
            Chem.RemoveHs(mol)

            msmi = MechSmiles(msmi_string)
            # msmi.standardize()

            # Needs an extra canonicalization as the ms.can_smiles is in Kekule
            same = Chem.MolToSmiles(mol) == Chem.MolToSmiles(
                Chem.MolFromSmiles(msmi.ms_prod.can_smiles)
            )
            # if not same:
            return same, ""
        except ReusedVirtualTSException:
            return False, "Virtual TS already used"
        except Exception as e:
            # print(f"Error with {msmi_string}: {e}")
            return False, str(e)
    else:
        return False, "No MechSMILES"


def _worker(args):
    # This code runs in a separate process → we can use SIGALRM safely on Unix.
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MAX_TIME)
    try:
        mech_smi, error = check_same_prod(args[0], args[1])
        return mech_smi, error
    except TaskTimeout:
        print(f"Timeout with {args = }")
        return False, f"Timeout {MAX_TIME}s"  # timed out → skip
    except Exception as e:
        # print(f"Exception {e} with {args}")
        return False, str(e)  # other failures → skip
    finally:
        signal.alarm(0)  # clear alarm


# Prep rows once (faster than iterrows)
rows = list(df[["elem_rxn", "mech_smi"]].itertuples(index=False, name=None))

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
    results = list(tqdm(ex.map(_worker, rows), total=len(rows)))

df["correct_prod"] = [res[0] if res else False for res in results]
df["error_same_prod"] = [res[1] if res else "" for res in results]

# df['correct_prod'] = df.parallel_apply(lambda x: check_same_prod(x['elem_rxn'], x['mech_smi']), axis=1)
# df['correct_prod'] = df.apply(lambda x: check_same_prod(x['elem_rxn'], x['mech_smi']), axis=1)

print(df[["correct_prod", "error_same_prod"]].value_counts())

df.to_csv(
    os.path.join("data", "flower_nature", "flower_new", f"{split}.csv"), index=None
)

check_same_prod(*df[["elem_rxn", "mech_smi"]].values[0])

# %% [markdown]
# Once everything is computed, we can merge them with a column indicating their splits

df_train = pd.read_csv(os.path.join("data", "flower_nature", "flower_new", "train.csv"))
df_val = pd.read_csv(os.path.join("data", "flower_nature", "flower_new", "val.csv"))
df_test = pd.read_csv(os.path.join("data", "flower_nature", "flower_new", "test.csv"))

df_train["split"] = "train"
df_val["split"] = "val"
df_test["split"] = "test"

df = pd.concat([df_train, df_val, df_test])

df["source_db"] = "flower_new"
df["source"] = df.apply(lambda x: f"{x['split']}_{x['rxn_idx']}", axis=1)
df["mech_smi_raw"] = df["mech_smi"].copy()
# df['mech_smi'] = df['mech_smi_raw'].parallel_apply(lambda x: MechSmiles(x).standard_value if not pd.isna(x) else '')


# %%
class TaskTimeout(Exception):  # noqa: F811 (redefinition of the class defined above)
    pass


def _timeout_handler(signum, frame):
    raise TaskTimeout()


def standardize_msmi(msmi_str):
    return (
        (MechSmiles(msmi_str).standard_value, "") if not pd.isna(msmi_str) else ("", "")
    )


def _worker(args):
    # This code runs in a separate process → we can use SIGALRM safely on Unix.
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MAX_TIME)
    try:
        mech_smi, error = standardize_msmi(args[0])
        return mech_smi, error
    except TaskTimeout:
        print(f"Timeout with {args = }")
        return "", f"Timeout {MAX_TIME}s"  # timed out → skip
    except Exception as e:
        # print(f"Exception {e} with {args}")
        return "", str(e)  # other failures → skip
    finally:
        signal.alarm(0)  # clear alarm


# Prep rows once (faster than iterrows)
rows = list(df[["mech_smi_raw"]].itertuples(index=False, name=None))

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
    results = list(tqdm(ex.map(_worker, rows), total=len(rows)))


df["mech_smi"] = [res[0] for res in results]
df["err_std"] = [res[1] for res in results]


df.to_csv(
    os.path.join("data", "flower_nature", "flower_new", "semi_final.csv"), index=None
)


# %%

df = pd.read_csv(os.path.join("data", "flower_nature", "flower_new", "semi_final.csv"))


# Careful, here we use source and not 'rxn_idx' as different splits use the same idx
# So we will renumber the index first
df["rxn_idx"] = pd.factorize(df["source"])[0]
df["step_idx_retro"] = df.groupby("rxn_idx").cumcount(ascending=False)

df[["rxn_idx", "step_idx_retro"]].head(20)

df_correct = df[df["correct_prod"]]
len(df_correct)


# Since we did some filtering, let's keep only complete sequences


def has_complete_distance_sequence(group):
    max_distance = group["step_idx_retro"].max()
    expected_distances = set(range(max_distance + 1))
    actual_distances = set(group["step_idx_retro"])
    return expected_distances == actual_distances


df_correct_filtered = df_correct.groupby("rxn_idx").filter(
    has_complete_distance_sequence
)


# %%


df_correct_filtered = df_correct_filtered.drop(
    columns=[
        "mech_smi_raw",
        "error_found",
        "error_same_prod",
        "err_std",
        "elem_rxn",
        "correct_prod",
    ]
)
df_correct_filtered.columns


df_correct_filtered.to_csv(
    os.path.join("data", "preprocessed", "flower_new_dataset.csv"), index=False
)


#
import pandas as pd

df_correct_filtered = pd.read_csv(
    os.path.join("data", "preprocessed", "flower_new_dataset.csv")
)
df_correct_train = df_correct_filtered[df_correct_filtered["split"] == "train"]
print(f"\nlen(df_correct_train) = {len(df_correct_train):_}")

df_correct_test = df_correct_filtered[df_correct_filtered["split"] == "test"]
print(f"\nlen(df_correct_test) = {len(df_correct_test):_}")
