# %% [markdown]
# This notebook will take a dataset composed of the following columns:
# mech_smi, rxn_idx, step_idx_retro, source, source_db
# Optionally, the csv can contain a column "split" to specify the train_val_test split
# And transforms it to a HuggingFace database fully ready for the training of a
# Mechanism agent in the ChRIMP framework.

# %% Imports
import os
from tqdm import tqdm
import pandas as pd
from collections import Counter

from chrimp.world.mechsmiles import MechSmiles

from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=22,
    progress_bar=True
)
# %%

#name = "mech_uspto_31k" # Name for the final HuggingFace DB
#path_csv = os.path.join("data", "preprocessed", "mech_uspto_31k.csv")

#name = "flower_dataset" # Name for the final HuggingFace DB
#path_csv = os.path.join("data", "preprocessed", "flower_new_dataset.csv")

#name = "pmechdb_elem" # Name for the final HuggingFace DB
#path_csv = os.path.join("data", "preprocessed", "pmechdb_split_0.csv")

#name = "uspto_ozonolysis" # Name for the final HuggingFace DB
#path_csv = os.path.join("data", "preprocessed", "USPTO_ozonolysis.csv")

name = "uspto_50_suzuki" # Name for the final HuggingFace DB
path_csv = os.path.join("data", "preprocessed", "USPTO_50_suzuki.csv")

# preprocessed df
df_pre = pd.read_csv(path_csv)
df_pre.columns

# %%
# Comment this block once the pipeline is ready
# Keep approx the 1k first lines
# df_pre = df_pre[df_pre['rxn_idx']<=288]

# %% Add length and forward index
dict_length = dict(df_pre.value_counts("rxn_idx"))

df_pre['rxn_length'] = df_pre['rxn_idx'].parallel_apply(lambda x: dict_length[x])
df_pre['step_idx_forward'] = df_pre.parallel_apply(lambda x: x['rxn_length']-x['step_idx_retro']-1, axis=1)

# If one needs to verify
#df_pre[['rxn_idx', 'rxn_length', 'step_idx_forward', 'step_idx_retro']].head(10)

# %% From the original MechSMILES, we can calculate quite fast the equilibrated

# But first let' s check if the dataset is already fully equilibrated
df_last_elem_step = df_pre[df_pre['step_idx_retro']==0]
df_last_elem_step['atom_counter'] = df_last_elem_step['mech_smi'].parallel_apply(lambda x: Counter([a.GetSymbol() for a in MechSmiles(x).ms.rdkit_mol.GetAtoms()]))
dict_atoms = dict(zip(df_last_elem_step['rxn_idx'].values,df_last_elem_step['atom_counter'].values))
df_pre['equ_with_last_step'] = df_pre.parallel_apply(lambda x: Counter([a.GetSymbol() for a in MechSmiles(x['mech_smi']).ms.rdkit_mol.GetAtoms()]) == dict_atoms[x['rxn_idx']], axis=1)
already_equilibrated = (df_pre['equ_with_last_step'].value_counts().get(False, 0)) == 0

# %%

if already_equilibrated:
    df_pre['mech_smi_equ'] = df_pre['mech_smi']
else:
    df_non_equ = df_pre[df_pre['equ_with_last_step'] == False]
    len(df_non_equ)
    df_non_equ[['rxn_idx', 'rxn_length', 'step_idx_retro', 'equ_with_last_step']]
    raise NotImplementedError("Implemenatation has been done for pmechdb, needs to be imported here")

def can_reac_prod_from_msmi(msmi: MechSmiles):
    return msmi.ms.can_smiles, msmi.ms_prod.can_smiles

df_pre['elem_reac_prod_equ_tuple'] = df_pre['mech_smi_equ'].parallel_apply(lambda x: can_reac_prod_from_msmi(MechSmiles(x)))
df_pre[['elem_reac_equ', 'elem_prod_equ']] = pd.DataFrame(df_pre['elem_reac_prod_equ_tuple'].tolist(), index=df_pre.index)
df_pre = df_pre.drop('elem_reac_prod_equ_tuple', axis=1)

df_pre.columns

# %% From the original MechSMILES, we can also calculate the minimal MechSMILES

def hide_cond_get_value(msmi: MechSmiles):
    msmi.hide_cond()
    return msmi.value

df_pre['mech_smi_min'] = df_pre['mech_smi_equ'].parallel_apply(lambda x: hide_cond_get_value(MechSmiles(x)))

df_pre['elem_reac_prod_min_tuple'] = df_pre['mech_smi_min'].parallel_apply(lambda x: can_reac_prod_from_msmi(MechSmiles(x)))
df_pre[['elem_reac_min', 'elem_prod_min']] = pd.DataFrame(df_pre['elem_reac_prod_min_tuple'].tolist(), index=df_pre.index)
df_pre = df_pre.drop('elem_reac_prod_min_tuple', axis=1)



## %% To avoid redoing all the calculations above and only calculate the species:
#from datasets import load_dataset
#ds_dict = load_dataset(f"SchwallerGroup/{name}")
#
#df_pre = pd.concat([
#    ds_dict['train'].to_pandas(),
#    ds_dict['val'].to_pandas(),
#    ds_dict['test'].to_pandas()
#], ignore_index=True)
#
#print(df_pre['split'].value_counts())


# %%
# From equilibrated, we can also caluculate fairly easily the "all species no stoichio." column
# by removing the duplicates

def create_spe_from_equ_column(df):
    """
    Add a "spe" column containing all unique species seen up to each index within each group.
    
    Parameters:
    df: DataFrame with columns 'elem_reac_spe', 'source', 'step_idx_forward'
    
    Returns:
    DataFrame with added 'elem_reac_spe' column
    """
    df = df.sort_values(['source', 'step_idx_forward']).copy()
    
    string2_list = []
    
    for group_name, group_df in tqdm(df.groupby('source', sort=False)):
        all_init_species = None

        
        for _, row in group_df.iterrows():
            # Split elem_reac_spe by '.' and add words to seen set
            species = set(row['elem_reac_equ'].split('.'))

            if all_init_species is None:
                all_init_species = species
                string2_list.append('.'.join(sorted(species)))

            else: # If not 1st step, we can use anything we have so far + the beginning species 
                string2_list.append('.'.join(sorted(all_init_species.union(species))))
    
    df['elem_reac_spe'] = string2_list
    
    return df


df_pre = create_spe_from_equ_column(df_pre)

# %%

# Also compute the value for the final goal of each 

df_pre = df_pre.rename(columns={"mech_smi":"mech_smi_ori"})
df_last_elem_step = df_pre[df_pre['step_idx_retro']==0]
dict_last_equ = dict(zip(df_last_elem_step['rxn_idx'].values,df_last_elem_step['elem_prod_equ'].values))
dict_last_min = dict(zip(df_last_elem_step['rxn_idx'].values,df_last_elem_step['elem_prod_min'].values))
#dict_last_spe = dict(zip(df_last_elem_step['rxn_idx'].values,df_last_elem_step['elem_prod_spe'].values))

df_pre['rxn_prod_equ'] = df_pre['rxn_idx'].parallel_apply(lambda x: dict_last_equ[x])
df_pre['rxn_prod_min'] = df_pre['rxn_idx'].parallel_apply(lambda x: dict_last_min[x])
#df_pre['rxn_prod_spe'] = df_pre['rxn_idx'].parallel_apply(lambda x: dict_last_spe[x])

# %% If not already split, make a split based on idx_single_step

if 'split' in df_pre.columns:
    train_df = df_pre[df_pre['split']=='train']
    val_df = df_pre[df_pre['split']=='val']
    test_df = df_pre[df_pre['split']=='test']

else:
    import random as rd

    all_rxn_indices = list(set(df_pre['rxn_idx']))
    num_single_steps = len(all_rxn_indices)
    rd.seed(22)
    rd.shuffle(all_rxn_indices)

    tvt_split = [0.8, 0.1, 0.1]
    assert sum(tvt_split) == 1, "The sum of tvt_split must add to 1"

    train_indices = all_rxn_indices[:int(num_single_steps*(tvt_split[0]))]
    val_indices = all_rxn_indices[int(num_single_steps*(tvt_split[0])):int(num_single_steps*(tvt_split[0]+tvt_split[1]))]
    test_indices = all_rxn_indices[int(num_single_steps*(tvt_split[0]+tvt_split[1])):]

    train_df = df_pre[df_pre['rxn_idx'].parallel_apply(lambda x: x in train_indices)]
    val_df = df_pre[df_pre['rxn_idx'].parallel_apply(lambda x: x in val_indices)]
    test_df = df_pre[df_pre['rxn_idx'].parallel_apply(lambda x: x in test_indices)]

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'


# %% Actually create the dataset
from datasets import Dataset, DatasetDict

train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_ds = Dataset.from_pandas(val_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

ds_dict = DatasetDict({
    "train": train_ds,
    "val": val_ds,
    "test": test_ds
})

ds_dict.push_to_hub(
    repo_id=f"SchwallerGroup/{name}",
    private=True,
    token=os.getenv("HF_TOKEN"),
)

# %% Load the dataset
import os
from datasets import load_dataset

# Available splits are "train", "val" and "test"
ds = load_dataset(f"SchwallerGroup/{name}", split = "train", token=os.getenv("HF_TOKEN"))
