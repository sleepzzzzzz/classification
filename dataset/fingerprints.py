import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

train = pd.read_csv('train.csv')
val = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')
all = pd.read_csv('hepatoprotection.csv')


def smile_to_morgan(file, filename):
    smi = file['smiles']
    mols = [Chem.MolFromSmiles(x) for x in smi]
    fingerprints = []
    safe = []
    labels = file['label']
    for mol_idx, mol in enumerate(mols):
        try:
            fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512)]
            fingerprints.append(fingerprint)
            safe.append(mol_idx)
        except:
            print("Error", mol_idx)
            continue

    np.save('{}_morgan.npy'.format(filename), fingerprints)
    np.save('{}_labels.npy'.format(filename), labels)


smile_to_morgan(train, 'train')
smile_to_morgan(val, 'val')
smile_to_morgan(test, 'test')
smile_to_morgan(all, 'all')
