"""Debug MMP Analysis"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from itertools import combinations

# Load a small sample
df = pd.read_csv('data/hERG.csv').head(100)
df = df[['SMILES', 'Name', 'pIC50']].dropna()

print(f"Loaded {len(df)} molecules")

# Fragment molecules
fragments_dict = {}
for idx, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol is None:
        continue
    
    frags = FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    print(f"\nMolecule {idx} ({row['Name']}): {len(frags)} fragmentations")
    
    for core, chains in frags:
        # Remove atom mapping
        mol_core = Chem.MolFromSmiles(core)
        mol_chains = Chem.MolFromSmiles(chains)
        
        if mol_core and mol_chains:
            for atom in mol_core.GetAtoms():
                atom.SetAtomMapNum(0)
            for atom in mol_chains.GetAtoms():
                atom.SetAtomMapNum(0)
            
            core_clean = Chem.MolToSmiles(mol_core)
            chains_clean = Chem.MolToSmiles(mol_chains)
            
            # Sort fragments
            chains_sorted = '.'.join(sorted(chains_clean.split('.')))
            
            print(f"  Core: {core_clean[:50]}")
            print(f"  Chains: {chains_sorted[:50]}")
            
            key = (core_clean, chains_sorted)
            if key not in fragments_dict:
                fragments_dict[key] = []
            fragments_dict[key].append({
                'smiles': row['SMILES'],
                'id': row['Name'],
                'activity': row['pIC50']
            })
    
    if idx > 5:  # Just test first few
        break

print(f"\n\nFound {len(fragments_dict)} unique core-chain combinations")
print(f"Combinations with 2+ molecules:")
multi_mols = {k: v for k, v in fragments_dict.items() if len(v) >= 2}
print(f"  {len(multi_mols)} combinations")

# Try to find MMPs
mmp_count = 0
for (core, chains), molecules in multi_mols.items():
    print(f"\nCore: {core[:50]}, Chains: {chains[:50]}")
    print(f"  Molecules: {len(molecules)}")
    
    for mol1, mol2 in combinations(molecules, 2):
        chains1 = chains.split('.')
        chains2 = chains.split('.')
        
        print(f"    Comparing {mol1['id']} vs {mol2['id']}")
        print(f"      Chains1: {chains1}")
        print(f"      Chains2: {chains2}")
        
        # This is the bug! Both mol1 and mol2 have the SAME chains because 
        # they're grouped by (core, chains). We need to look at the ORIGINAL
        # molecule's fragmentation, not the grouped chains!
        
print(f"\nTotal MMPs found: {mmp_count}")
