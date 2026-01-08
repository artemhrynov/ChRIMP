# %% Imports
import pandas as pd
import os
from tqdm import tqdm
import signal

from chrimp.world.mechsmiles import MechSmiles
from chrimp.world.molecule_set import (
    MoleculeSet,
    RadicalAtomException,
    ReusedVirtualTSException,
)

from chrimp.dataset.pmechdb_helper import translate_and_validate

from concurrent.futures import ProcessPoolExecutor
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=15, progress_bar=True)

# In the scope of this work, we want to disallow radicals for the time being
MoleculeSet.default_authorize_radicals = False
MoleculeSet.default_treat_P_S_Cl_12_electrons = False

MAX_TIME = 10  # seconds per task
MAX_WORKERS = 20

# %%

# %% Load data

splits = ["train", "val", "test"]
df_splits = []
last_index = 0

for split in splits:
    # Data retrieved in https://github.com/rjmille3/ArrowFinder (accessed Nov 3, 2025). One split is available under MIT license
    df_raw = pd.read_csv(
        os.path.join(
            "data", "pmechdb", "mc_train_fold0", "reformatted", f"{split}.txt"
        ),
        names=["smirks", "arrow_code", "source", "sink"],
    )

    df_raw["pmechdb_str"] = df_raw.apply(
        lambda x: f"{x['smirks']} {x['arrow_code']}", axis=1
    )
    df_raw["split"] = split
    df_raw["source_db"] = "pmechdb_split_0"
    df_raw["source"] = [f"{split}_{i}" for i in range(len(df_raw))]
    df_raw["step_idx_retro"] = 0  # These data are elementary steps
    df_splits.append(df_raw)

# %%
df = pd.concat(df_splits)
df["rxn_idx"] = range(len(df))
df["split"].value_counts()
len_df_all = len(df)

# Remove trivial elementary steps (effectively acting as stop tokens for diffusion models probably)
df = df[df["smirks"].apply(lambda x: x.split(">")[0] != x.split(">")[-1])]
len_df_digit_non_trivial = len(df)

print(f"Len of df                    : {len_df_all:_}")
print(
    f"And non-trivial              : {len_df_digit_non_trivial:_} ({len_df_digit_non_trivial/len_df_all:.2%} of previous)"
)


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
        mech_smi, error = translate_and_validate(args[0])
        return mech_smi, error

    except TaskTimeout:
        print(f"Timeout with {args = }")
        return "", f"Timeout {MAX_TIME}s"  # timed out → skip

    except ReusedVirtualTSException:
        return "", "Virtual TS reused"

    except RadicalAtomException:
        return "", "Radical species"

    except AttributeError:
        return "", "Invalid SMILES"

    except Exception as e:
        print(f"Exception {e} with {args}")
        return "", str(e)  # other failures → skip
    finally:
        signal.alarm(0)  # clear alarm


# Prep rows once (faster than iterrows)
rows = list(df[["pmechdb_str"]].itertuples(index=False, name=None))

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

# %% Look at the errors

# 144 involve radical, which is a current limitation of the code and needs more testing
print(df[df["error_found"] == "Radical species"]["pmechdb_str"].values[:5])

# 5 species trigger canonicalization issues with hypervalent Br-
print(df[df["error_found"] == "Invalid SMILES"]["pmechdb_str"].values)


# %% MechSmiles canonicalization

df["mech_smi_raw"] = df["mech_smi"].copy()


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

df.to_csv(os.path.join("data", "pmechdb", "semi_final.csv"), index=None)


# %%
df_correct_filtered = df[
    df.apply(lambda x: x["err_std"] == "" and x["error_found"] == "", axis=1)
]
df_correct_filtered = df_correct_filtered.drop(
    columns=[
        "mech_smi_raw",
        "error_found",
        "err_std",
        "smirks",
        "arrow_code",
        "pmechdb_str",
        "source",
        "sink",
    ]
)
df_correct_filtered.columns


df_correct_filtered.to_csv(
    os.path.join("data", "preprocessed", "pmechdb_split_0.csv"), index=False
)
