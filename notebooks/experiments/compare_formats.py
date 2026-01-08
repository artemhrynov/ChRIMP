# %% [markdown]
# # 1st experiment, char-efficiency of MechSMILES
# The conventions go as follow:
#
# | Convention               | MechSMILES     | PMechDB     | mech-USPTO-31k                                      |
# |:------------------------:|:--------------:|:-----------:|:---------------------------------------------------:|
# | Attack                   | (a,b)          | "a=b"       | (a,b) if bond-order == 0 else (a,[a,b])             |
# | Ionization               | ((a,b), b)     | "a,b=b"     | ([a,b], b)                                          |
# | Bond-attack              | ((a,b), c)     | "b,a=b,c"   | ([a,b], c) if bond-order == 0 else ([a,b], [b,c])   |
# | Homo-cleavage            | (hv, (a,b))    | ?           | ?                                                   |

# %% imports
import os
import re
import pandas as pd
from datasets import load_dataset

ds = load_dataset("SchwallerGroup/pmechdb_elem", split="train")
df_all = pd.read_csv(os.path.join("data", "pmechdb", "semi_final.csv"))
df = df_all[df_all.apply(lambda x: x['split']=="train" and pd.isna(x['err_std']) and pd.isna(x['error_found']), axis=1)].reset_index()


print(f"{(len(ds) == len(df)) = }")


# %% Translate to mech-USPTO-31k format
ds_df = ds.to_pandas()

from chrimp.world.mechsmiles import MechSmiles
from rdkit.Chem import MolFromMolBlock, MolFromSmiles, MolToSmiles
from ast import literal_eval


def translate_arrows(arrows, attacks_on_bonds):
    new_arrows = []

    for a_str, aob in zip(arrows, attacks_on_bonds):
        a = literal_eval(a_str)
        if not aob:
            if isinstance(a[0], int):
                new_arrows.append(a_str.replace(' ', ''))

            else: # If not an atom-attack, we rewrite the bond in the correct format
                a_str = a_str.replace(' ', '')

                # Replace '(' that's not at the start of the string
                a_str = re.sub(r'(?<!^)\(', '[', a_str)

                # Replace ')' that's not at the end of the string  
                a_str = re.sub(r'\)(?!$)', ']', a_str)

                new_arrows.append(a_str)

        elif isinstance(a[1], int): # Attack on a bond
            new_arrows.append(f"({a[0]},[{a[0]},{a[1]}])")
        else: # Bond-attack on bond
            new_arrows.append(f"([{a[0]},{a[1]}],[{a[1]},{a[2]}])")
        
    return new_arrows

def tag_mapped_Hs_and_remap(smiles) -> tuple[dict, str]:
    """
        Returns a list of all index of Hs attached to a heavy atom, and remap the molecule
        to give
    """


    mol = MolFromSmiles(smiles, sanitize=False)
    remapping_dict = dict()

    for a in mol.GetAtoms():
        if a.GetSymbol() == "H" and (map_h := a.GetAtomMapNum()) > 0:
            if len(a.GetBonds()) > 1:
                raise ValueError("Hydrogen with more than one bond?")

            other_atom = a.GetBonds()[0].GetOtherAtom(a)
            if other_atom.GetSymbol() != "H": # Only annoying case would be H2, but we need to check for that

                if (other_map := other_atom.GetAtomMapNum()) != 0:
                    remapping_dict[map_h] = f"{other_map}.1"
                    a.SetAtomMapNum(0)

                else:
                    remapping_dict[map_h] = f"{map_h}.1"
                    other_atom.SetAtomMapNum(map_h)
                    a.SetAtomMapNum(0)

    return (
        remapping_dict,
        MolToSmiles(mol),
    )

def map_remaining_heavy(smiles):
    mol = MolFromSmiles(smiles)
    idx_mapping_pattern = r":(\d+)]"
    already_used_indices = {int(match) for match in re.findall(idx_mapping_pattern, smiles)}
    biggest_idx_so_far = 0

    for a in mol.GetAtoms():
        while biggest_idx_so_far + 1 in already_used_indices:
            biggest_idx_so_far += 1

        if a.GetAtomMapNum() == 0:
            a.SetAtomMapNum(biggest_idx_so_far + 1)
            biggest_idx_so_far += 1

    return MolToSmiles(mol)

def remap_arrow(tup_str, remapping_dict, depth: int = 0):

    tup = literal_eval(tup_str)

    if isinstance(tup, int):
        return float(remapping_dict[tup]) if tup in remapping_dict.keys() else tup
    
    elif isinstance(tup, tuple):
        if depth == 0:
            return str(tuple(remap_arrow(str(sub_tup), remapping_dict, depth = depth+1) for sub_tup in tup))
        else:
            return tuple(remap_arrow(str(sub_tup), remapping_dict, depth = depth+1) for sub_tup in tup)

    elif isinstance(tup, list):
        return [remap_arrow(str(sub_tup), remapping_dict, depth = depth+1) for sub_tup in tup]


def mechsmiles_to_mapped_elementary(msmi_str):
    msmi = MechSmiles(msmi_str)
    # For each attack or bond attack arrow, detect if the bond order is already >= 1
    # as mech-USPTO-31k format treats differently if attacks on an atom with which
    # the attacks shares a bond yet or not.
    molecule_set = msmi.ms
    arrows_list = msmi.smiles_arrows

    attacks_on_bonds = molecule_set.tag_attacks_on_bonds([msmi.process_smiles_arrow(a, molecule_set.atom_map_dict) for a in msmi.smiles_arrows])

    intermediate_arrows = translate_arrows(arrows_list, attacks_on_bonds)

    remap_dict, new_smiles = tag_mapped_Hs_and_remap(msmi_str.split('|')[0])

    # remap the arrow with the implicit H notation and map every heavy remaining atom
    final_smiles = map_remaining_heavy(new_smiles)

    final_arrows = [remap_arrow(arr, remap_dict) for arr in intermediate_arrows]
    # We add a separator for smiles and arrows, representing the two objects we receive
    # in the format of mech-USPTO-31k
    return f"{final_smiles}|{','.join(final_arrows)}"

ds_df['mech_upsto_31k_format'] = ds_df['mech_smi_equ'].apply(lambda x: mechsmiles_to_mapped_elementary(x))

# %% Calculate the mapped elementary steps as well (Similar to FlowER)

def mechsmiles_to_mapped_elementary(msmi_str, removeHs=False):
    msmi = MechSmiles(msmi_str)
    mol_reac = MolFromMolBlock(msmi.ms.molblock, removeHs = removeHs)
    mol_prod = MolFromMolBlock(msmi.ms_prod.molblock, removeHs = removeHs)
    for i, (a_r, a_p) in enumerate(zip(mol_reac.GetAtoms(), mol_prod.GetAtoms())):
        a_r.SetAtomMapNum(i+1)
        a_p.SetAtomMapNum(i+1)
    return f"{MolToSmiles(mol_reac)}>>{MolToSmiles(mol_prod)}"

ds_df['mapped_elem_w_hs'] = ds_df['mech_smi_equ'].apply(lambda x: mechsmiles_to_mapped_elementary(x))
ds_df['mapped_elem_wo_hs'] = ds_df['mech_smi_equ'].apply(lambda x: mechsmiles_to_mapped_elementary(x, removeHs=True))

# %% print distribution for the different metrics

# I will remove white spaces in MechSMILES, are they are neither generated
# at inference, nor necessary for storage

# Calculate character lengths
df['pmechdb_str_len'] = df['pmechdb_str'].apply(lambda x: len(str(x)))
ds_df['mech_smi_min_len'] = ds_df['mech_smi_min'].apply(lambda x: len(str(x.replace(' ', ''))))
ds_df['mech_smi_equ_len'] = ds_df['mech_smi_equ'].apply(lambda x: len(str(x.replace(' ', ''))))
ds_df['mapped_elem_w_hs_len'] = ds_df['mapped_elem_w_hs'].apply(lambda x: len(str(x)))
ds_df['mapped_elem_wo_hs_len'] = ds_df['mapped_elem_wo_hs'].apply(lambda x: len(str(x)))
ds_df['mech_upsto_31k_format_len'] = ds_df['mech_upsto_31k_format'].apply(lambda x: len(str(x))) 

# %% Create seaborn plots
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Prepare data for plotting
plot_data = pd.DataFrame({
    'pmechdb_str': df['pmechdb_str_len'],
    'mech_smi_min': ds_df['mech_smi_min_len'],
    'mech_smi_equ': ds_df['mech_smi_equ_len'],
    'mech_uspto': ds_df['mech_upsto_31k_format_len'],
    'mapped_elem_w_hs': ds_df['mapped_elem_w_hs_len'],
    'mapped_elem_wo_hs': ds_df['mapped_elem_wo_hs_len']
})

# Melt the dataframe for easier plotting
plot_data_melted = plot_data.melt(var_name='Representation', value_name='Character Length')

## Create figure with multiple subplots
#fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#
## 1. Box plot
#sns.boxplot(data=plot_data_melted, x='Representation', y='Character Length', ax=axes[0, 0])
#axes[0, 0].set_title('Box Plot: Character Length Distribution')
#axes[0, 0].set_xlabel('Representation')
#axes[0, 0].set_ylabel('Character Length')
#
## 2. Violin plot
#sns.violinplot(data=plot_data_melted, x='Representation', y='Character Length', ax=axes[0, 1])
#axes[0, 1].set_title('Violin Plot: Character Length Distribution')
#axes[0, 1].set_xlabel('Representation')
#axes[0, 1].set_ylabel('Character Length')
#
## 3. Histogram with KDE
#for col in plot_data.columns:
#    sns.histplot(plot_data[col], kde=True, label=col, alpha=0.5, ax=axes[1, 0])
#axes[1, 0].set_title('Histogram with KDE: Character Length Distribution')
#axes[1, 0].set_xlabel('Character Length')
#axes[1, 0].set_ylabel('Frequency')
#axes[1, 0].legend()
#
## 4. Bar plot with mean values
#mean_values = plot_data.mean()
#sns.barplot(x=mean_values.index, y=mean_values.values, ax=axes[1, 1])
#axes[1, 1].set_title('Mean Character Length Comparison')
#axes[1, 1].set_xlabel('Representation')
#axes[1, 1].set_ylabel('Mean Character Length')
#for i, v in enumerate(mean_values.values):
#    axes[1, 1].text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
#
#plt.tight_layout()
#plt.show()

# %% Horizontal box plot
fig, ax = plt.subplots(figsize=(15, 6))

# Create a copy with renamed labels for the legend
plot_data_renamed = plot_data_melted.copy()
label_mapping = {
    'mapped_elem_w_hs': 'Mapped elementary (with Hs)\n (FlowER format)',
    'mapped_elem_wo_hs': 'Mapped elementary (without Hs)',
    'mech_uspto':  "Chen's alternative to MechSMILES\n(mech-USPTO-31k format)",
    'pmechdb_str': 'SMIRKS + arrow-code\n(PMechDB format)',
    'mech_smi_equ': 'equilibrated MechSMILES',
    'mech_smi_min': 'minimal MechSMILES',
}
plot_data_renamed['Representation'] = plot_data_renamed['Representation'].map(label_mapping)

# Convert to categorical with the order from the dictionary
plot_data_renamed['Representation'] = pd.Categorical(
    plot_data_renamed['Representation'],
    categories=list(label_mapping.values()),
    ordered=True
)

sns.boxplot(data=plot_data_renamed, y='Representation', x='Character Length', ax=ax, palette="plasma")
ax.set_title('Character Length Distribution of the training split of PMechDB split 0')
ax.set_ylabel('Representation')
ax.set_xlabel('Character Length')
plt.tight_layout()
plt.savefig(os.path.join("data", "figures", "paper_experiments", "comparison_char_efficiency.svg"))
plt.show()

# %% Horizontal box plot (zoomed)
fig, ax = plt.subplots(figsize=(15, 4))

# Create a copy with renamed labels for the legend
plot_data_renamed = plot_data_melted.copy()
label_mapping = {
    'mapped_elem_w_hs': 'Mapped elementary (with Hs)\n FlowER format',
    'mapped_elem_wo_hs': 'Mapped elementary (without Hs)',
    'mech_uspto':  "Chen's alternative to MechSMILES\n(mech-USPTO-31k format)",
    'pmechdb_str': 'SMIRKS + arrow-code\nPMechDB format',
    'mech_smi_equ': 'equilibrated MechSMILES\nOurs (storage)',
    'mech_smi_min': 'minimal MechSMILES\nOurs (inference)',
}
plot_data_renamed['Representation'] = plot_data_renamed['Representation'].map(label_mapping)

# Convert to categorical with the order from the dictionary
plot_data_renamed['Representation'] = pd.Categorical(
    plot_data_renamed['Representation'],
    categories=list(label_mapping.values()),
    ordered=True
)

sns.boxplot(data=plot_data_renamed, y='Representation', x='Character Length', ax=ax, palette="plasma")

# Create legend instead of y-axis labels
colors = sns.color_palette("plasma", n_colors=len(label_mapping))
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=label)
                   for i, label in enumerate(label_mapping.values())]
ax.legend(handles=legend_elements, loc='lower right')

# Remove y-axis labels
ax.set_yticklabels([])
ax.set_ylabel('')

ax.set_title('Character Length Distribution of the training split of PMechDB split 0')
ax.set_xlabel('Character Length')
ax.set_xlim(left=0, right=1000)
plt.tight_layout()
plt.savefig(os.path.join("data", "figures", "paper_experiments", "comparison_char_efficiency_zoom.svg"))
plt.show()

# %% Print summary statistics
print("=" * 60)
print("Summary Statistics:")
print("=" * 60)
print(plot_data.describe())
print(f"\nCharacter efficiency (vs pmechdb_str):")
print(f"  mech_smi_min: {(mean_values['mech_smi_min'] / mean_values['pmechdb_str'] * 100):.2f}%")
print(f"  mech_smi_equ: {(mean_values['mech_smi_equ'] / mean_values['pmechdb_str'] * 100):.2f}%")
print(f"  mapped_elem_w_hs: {(mean_values['mapped_elem_w_hs'] / mean_values['pmechdb_str'] * 100):.2f}%")
print(f"  mapped_elem_wo_hs: {(mean_values['mapped_elem_wo_hs'] / mean_values['pmechdb_str'] * 100):.2f}%")

