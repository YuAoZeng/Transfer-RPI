import pandas as pd
import torch
from PyBioMed.PyProtein import PseudoAAC

# Polarity
polarity = {
    "A": -0.591, "C": -1.343, "D": 1.05, "E": 1.357, "F": -1.006,
    "G": -0.384, "H": 0.336, "I": -1.239, "K": 1.831, "L": -1.019,
    "M": -0.663, "N": 0.945, "P": 0.189, "Q": 0.931, "R": 1.538,
    "S": -0.228, "T": -0.032, "V": -1.337, "W": -0.595, "Y": 0.26
}

# Secondary structure
secondary_structure = {
    "A": -1.302, "C": 0.465, "D": 0.302, "E": -1.453, "F": -0.59,
    "G": 1.652, "H": -0.417, "I": -0.547, "K": -0.561, "L": -0.987,
    "M": -1.524, "N": 0.828, "P": 2.081, "Q": -0.179, "R": -0.055,
    "S": 1.399, "T": 0.326, "V": -0.279, "W": 0.009, "Y": 0.83
}

# Molecular volume
molecular_volume = {
    "A": -0.733, "C": -0.862, "D": -3.656, "E": 1.477, "F": 1.891,
    "G": 1.33, "H": -1.673, "I": 2.131, "K": 0.533, "L": -1.505,
    "M": 2.219, "N": 1.299, "P": -1.628, "Q": -3.005, "R": 1.502,
    "S": -4.76, "T": 2.213, "V": -0.544, "W": 0.672, "Y": 3.097
}

# Codon diversity
codon_diversity = {
    "A": 1.57, "C": -1.02, "D": -0.259, "E": 0.113, "F": -0.397,
    "G": 1.045, "H": -1.474, "I": 0.393, "K": -0.277, "L": 1.266,
    "M": -1.005, "N": -0.169, "P": 0.421, "Q": -0.503, "R": 0.44,
    "S": 0.67, "T": 0.908, "V": 1.242, "W": -2.128, "Y": -0.838
}

# Electrostatic charge
electrostatic_charge = {
    "A": -0.146, "C": -0.255, "D": -3.242, "E": -0.837, "F": 0.412,
    "G": 2.064, "H": -0.078, "I": 0.816, "K": 1.648, "L": -0.912,
    "M": 1.212, "N": 0.933, "P": -1.392, "Q": -1.853, "R": 2.897,
    "S": -2.647, "T": 1.313, "V": -1.262, "W": -0.184, "Y": 1.512
}


def compute_paac_features(protein_seq, lambdaValue=25, w=0.05):
    properties = {
        "Hydrophobicity": PseudoAAC._Hydrophobicity,
        "Hydrophilicity": PseudoAAC._hydrophilicity,
        "ResidueMass": PseudoAAC._residuemass,
        "Polarity": polarity,
        "Secondary structure": secondary_structure,
        "Molecular volume": molecular_volume,
        "Codon diversity": codon_diversity,
        "Electrostatic charge": electrostatic_charge
    }

    combined_features_vector = []
    for prop_values in properties.values():
        paac_features = PseudoAAC.GetPseudoAAC(protein_seq, lamda=lambdaValue, weight=w, AAP=[prop_values])
        combined_features_vector.extend(paac_features.values())

    return combined_features_vector


def load_and_process_csv(filepath, lambdaValue=25, w=0.05):
    # 定义标准氨基酸字母表
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    df = pd.read_csv(filepath)
    results = []

    for index, row in df.iterrows():
        protein_id = row['protein_id']
        protein_seq = row['protein_seq']


        # 过滤掉非标准氨基酸
        filtered_protein_seq = ''.join([aa for aa in protein_seq if aa in standard_amino_acids])

        # 计算 PAAC 特征
        paac_features_vector = compute_paac_features(filtered_protein_seq, lambdaValue=lambdaValue, w=w)


        torch.save(torch.tensor(paac_features_vector), f'../data/RPI488/protein_PAAC/{protein_id}.pt')

        results.append(f'{protein_id}.pt')

    return results


processed_files = load_and_process_csv('../data/RPI488/RPI488_proteins.csv')

print(len(processed_files))

