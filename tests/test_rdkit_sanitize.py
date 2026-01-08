# I need a piece of code that:
# - does the difference between smiles 0 and smiles 1
# - converts smiles 2 to its kekule form
# - doesn't add hydrogens on smiles 3
# - remove explicit unmapped hydrogens on smiles 4
from rdkit import Chem

smiles_list = [
    "C1=CC=CC(CC)=C1C",
    "C1=CC=C(CC)C(C)=C1",
    "c1cccc(CC)c1C",
    "[C]=O",
    "[H]C([H])=O"
]

print("Normal snippet")
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    new_smiles = Chem.MolToSmiles(mol)
    print(f'new_smiles = "{new_smiles}"')

print("Custom snippet")
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    new_smiles = Chem.MolToSmiles(mol)

    print(f'new_smiles = "{new_smiles}"')





