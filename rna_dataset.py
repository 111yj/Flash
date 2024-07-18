import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
nucleotide_to_kmer = {
    'A': 1, 'G': 2, 'C': 3, 'U': 4, 'N': 9,
    'AA': 11, 'AG': 12, 'AC': 13, 'AU': 14, 'AN': 19,
    'GA': 21, 'GG': 22, 'GC': 23, 'GU': 24, 'GN': 29,
    'CA': 31, 'CG': 32, 'CC': 33, 'CU': 34, 'CN': 39,
    'UA': 41, 'UG': 42, 'UC': 43, 'UU': 44, 'UN': 49,
    'NA': 91, 'NG': 92, 'NC': 93, 'NU': 94, 'NN': 99,
    'AAA': 111, 'AAG': 112, 'AAC': 113, 'AAU': 114,'AAN': 119,
    'AGA': 121, 'AGG': 122, 'AGC': 123, 'AGU': 124,'AGN': 129,
    'ACA': 131, 'ACG': 132, 'ACC': 133, 'ACU': 134,'ACN': 139,
    'AUA': 141, 'AUG': 142, 'AUC': 143, 'AUU': 144,'AUN': 149,
    'GAA': 211, 'GAG': 212, 'GAC': 213, 'GAU': 214,'GAN': 219,
    'GGA': 221, 'GGG': 222, 'GGC': 223, 'GGU': 224,'GGN': 229,
    'GCA': 231, 'GCG': 232, 'GCC': 233, 'GCU': 234,'GCN': 239,
    'GUA': 241, 'GUG': 242, 'GUC': 243, 'GUU': 244,'GUN': 249,
    'CAA': 311, 'CAG': 312, 'CAC': 313, 'CAU': 314,'CAN': 319,
    'CGA': 321, 'CGG': 322, 'CGC': 323, 'CGU': 324,'CGN': 329,
    'CCA': 331, 'CCG': 332, 'CCC': 333, 'CCU': 334,'CCN': 339,
    'CUA': 341, 'CUG': 342, 'CUC': 343, 'CUU': 344,'CUN': 349,
    'UAA': 411, 'UAG': 412, 'UAC': 413, 'UAU': 414,'UAN': 419,
    'UGA': 421, 'UGG': 422, 'UGC': 423, 'UGU': 424,'UGN': 429,
    'UCA': 431, 'UCG': 432, 'UCC': 433, 'UCU': 434,'UCN': 439,
    'UUA': 441, 'UUG': 442, 'UUC': 443, 'UUU': 444,'UUN': 449,
    'NAA': 911, 'NAG': 912, 'NAC': 913, 'NAU': 914,'NAN': 919,
    'NGA': 921, 'NGG': 922, 'NGC': 923, 'NGU': 924,'NGN': 929,
    'NCA': 931, 'NCG': 932, 'NCC': 933, 'NCU': 934,'NCN': 939,
    'NUA': 941, 'NUG': 942, 'NUC': 943, 'NUU': 944,'NUN': 949,
    'NNA': 991, 'NNG': 992, 'NNC': 993, 'NNU': 994,'NNN': 999,
}

my_embedding_dim = 50 #24-64
my_embedding_layer = torch.nn.Embedding(512, 100)

class GeneDataset(Dataset):#数据集类 将基因序列转换为k-mer序列，为每个样本分配标签
    def __init__(self, gene_data, nucleotide_to_kmer, kmer_lengths):
        self.gene_data = gene_data#列表 每个元素包含一个基因序列和相应的标签
        self.nucleotide_to_kmer = nucleotide_to_kmer#一个字典 将aucg映射到对应的k-mer编码
        self.kmer_lengths = kmer_lengths #一个包含k-mer长度的列表
        self.embedding_dim = my_embedding_dim
        self.embedding_layer = my_embedding_layer

    def __len__(self):
        return len(self.gene_data)

    def __getitem__(self, index):#提取基因序列和标签
        data_item = self.gene_data[index]
        gene_sequence = data_item['sequence']#基因序列
        label = data_item['label']#标签

        # 构造k-mer序列
        channels = []
        for kmer_length in self.kmer_lengths:
            kmers = [gene_sequence[i:i+kmer_length] for i in range(len(gene_sequence)-kmer_length+1)]
            encoded_sequence = torch.tensor([self.nucleotide_to_kmer[kmer] for kmer in kmers], dtype=torch.long)
            embedded_sequence = my_embedding_layer(encoded_sequence)
            padded_sequence = F.pad(embedded_sequence, (0, self.embedding_dim - embedded_sequence.size(1), 0, self.embedding_dim - embedded_sequence.size(0)))
            channels.append(padded_sequence.unsqueeze(0))

        channels = torch.cat(channels, dim=0)
        return channels, label





def load_samples_with_labels(file_path):
    samples_with_labels = []
    current_label = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            if line.startswith('>+sample'):
                current_label = 1  # 正样本标签为1
            elif line.startswith('>-sample'):
                current_label = 0  # 负样本标签为0
            else:

                sequence = line
                if current_label is not None:
                    # 只有在设置了当前标签后才添加序列
                    samples_with_labels.append({'sequence': sequence, 'label': current_label})
                else:
                    # 如果在序列之前没有遇到标记，则打印错误信息
                    print(f"无法识别的行，缺少标签: {line}")

    return samples_with_labels


#新


def preprocessData(data_file):
    # 加载数据
    gene_data = load_samples_with_labels(data_file)

    # 创建 GeneDataset 实例
    gene_dataset = GeneDataset(gene_data, nucleotide_to_kmer, [1, 2, 3])

    # 将数据分成正样本和负样本
    positive_samples = []
    negative_samples = []
    for index in range(len(gene_dataset)):
        sample, label = gene_dataset[index]
        if label == 1:
            positive_samples.append(sample)
        elif label == 0:
            negative_samples.append(sample)

    # 分别划分正样本和负样本的训练集和测试集
    pos_train, pos_test = train_test_split(positive_samples, test_size=0.5, random_state=42)
    neg_train, neg_test = train_test_split(negative_samples, test_size=0.5, random_state=42)

    # 合并训练集和测试集
    train_gene_data = pos_train + neg_train
    test_gene_data = pos_test + neg_test

    return train_gene_data, test_gene_data


def main():
    data_file=r'C:\Users\lenovo\Desktop\model\iRNA-m6A\iRNA-m6A\benchmark\benchmark\h_b_all.fa'
    train_gene_data, test_gene_data = preprocessData(data_file)
    torch.save(train_gene_data, 'gene_train.pth')
    torch.save(test_gene_data, 'gene_test.pth')#将训练集和测试集的数据保存到名为 'gene_train.pth' 和 'gene_test.pth' 的文件中。



if __name__ == '__main__':
    main()