from rdkit.Chem import MolFromSmiles, MolToSmiles


def quick_canonicalize(smiles, sanitize: bool = False):
    return MolToSmiles(MolFromSmiles(smiles, sanitize=sanitize))
