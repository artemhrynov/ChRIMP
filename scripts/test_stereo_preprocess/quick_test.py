from rdkit import Chem

pred = "COC(=O)C1=CC(C2=[NH+]C(O)(C3=CC=C(C(F)(F)F)C=C3)CS2)=CC=C1"
target = "COC(=O)C1=CC(C2=[N+:1]([H:2])C(O)(C3=CC=C(C(F)(F)F)C=C3)CS2)=CC=C1.[Br-:3]"

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