from repDNA.nac import Kmer
import pandas as pd
import torch
import re


"""
If you encounter the following error:
    from nacutil import make_upto_kmer_list, make_revcomp_kmer_list, make_kmer_vector
ModuleNotFoundError: No module named 'nacutil'
    from util import get_data
ModuleNotFoundError: No module named 'util'
    from util import frequency
ModuleNotFoundError: No module named 'util'

Please replace the following import statement:
    from nacutil import make_upto_kmer_list, make_revcomp_kmer_list, make_kmer_vector
    from util import get_data
    from util import frequency
with:
    from .nacutil import make_upto_kmer_list, make_revcomp_kmer_list, make_kmer_vector
    from .util import get_data
    from .util import frequency
"""

def clean_rna(sequence):
    # 删除所有非 A、G、C、T、U 的字符
    cleaned_sequence = re.sub('[^AGCTU]', '', sequence)

    # 将所有的 U 替换为 T
    modified_sequence = re.sub('U', 'T', cleaned_sequence)

    return modified_sequence

# 定义k-mer处理对象
kmer = Kmer(k=4, normalize=True, upto=True)

# 读取CSV文件
df = pd.read_csv('../data/RPI2241/RPI2241_rnas.csv')

# 遍历每行数据
for index, row in df.iterrows():
    rna_seq = row['RNA_seq']
    rna_seq = clean_rna(rna_seq)
    rna_id = row['RNA_id']

    # 计算k-mer特征
    aptamer_kmer = kmer.make_kmer_vec([rna_seq])

    # 将结果保存为.pt文件
    torch.save(aptamer_kmer[0], f'../data/RPI2241/rna_kmer/{rna_id}.pt')

print("save")
