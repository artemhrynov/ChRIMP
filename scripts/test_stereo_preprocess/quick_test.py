from rdkit import Chem

pred = "C=CC([O-])(Cl)[NH+]1CCC[C@H]1C(=O)OC"
target = "C=C[C:1]([NH+]1CCCC1C(=O)OC)([O-:2])[Cl:3]"

def can(smiles, stereo=True):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=stereo, canonical=True)

print("With stereo:")
print(can(pred, stereo=True))
print(can(target, stereo=True))

print("\nWithout stereo:")
print(can(pred, stereo=False))
print(can(target, stereo=False))