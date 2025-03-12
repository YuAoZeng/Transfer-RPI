import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from main import calc_metrics, load_pre_train_representations, get_hyperparameter


class Transfer_RPI(nn.Module):
    def __init__(self):
        super(Transfer_RPI, self).__init__()

        # RNA：Bi-LSTM
        self.rna_lstm = nn.LSTM(input_size=1280, hidden_size=320, batch_first=True, bidirectional=True)
        self.rna_fc = nn.Linear(640, 1280)

        # 蛋白质：Bi-LSTM
        self.protein_lstm = nn.LSTM(input_size=1280, hidden_size=320, batch_first=True, bidirectional=True)
        self.protein_fc = nn.Linear(640, 1280)

        # 特征融合预测
        self.fc1 = nn.Linear(1280 * 2, 1280)
        self.fc2 = nn.Linear(1280, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, rna_embedding, protein_embedding):
        # RNA 嵌入特征提取
        rna_embedding = rna_embedding.unsqueeze(1)
        rna_lstm_out, _ = self.rna_lstm(rna_embedding)
        rna_out = F.relu(self.rna_fc(rna_lstm_out[:, -1, :]))  # 使用最后一个时间步的隐藏状态

        # 蛋白质嵌入特征提取
        protein_embedding = protein_embedding.unsqueeze(1)
        protein_lstm_out, _ = self.protein_lstm(protein_embedding)
        protein_out = F.relu(self.protein_fc(protein_lstm_out[:, -1, :]))

        # 融合 RNA 和蛋白质特征
        concatenated = torch.cat((rna_out, protein_out), dim=1)

        # 预测
        x = F.relu(self.fc1(concatenated))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()


class Transfer_RPI2(nn.Module):
    def __init__(self):
        super(Transfer_RPI2, self).__init__()

        self.rna_lstm = nn.LSTM(input_size=340, hidden_size=320, batch_first=True, bidirectional=True)
        self.rna_fc = nn.Linear(640, 1280)

        self.protein_lstm = nn.LSTM(input_size=360, hidden_size=320, batch_first=True, bidirectional=True)
        self.protein_fc = nn.Linear(640, 1280)

        self.fc1 = nn.Linear(1280 * 2, 1280)
        self.fc2 = nn.Linear(1280, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, ran_kmer, protein_PAAC):
        rna_embedding = ran_kmer.float().unsqueeze(1)
        rna_lstm_out, _ = self.rna_lstm(rna_embedding)
        rna_out = F.relu(self.rna_fc(rna_lstm_out[:, -1, :]))

        protein_embedding = protein_PAAC.unsqueeze(1)
        protein_lstm_out, _ = self.protein_lstm(protein_embedding)
        protein_out = F.relu(self.protein_fc(protein_lstm_out[:, -1, :]))

        concatenated = torch.cat((rna_out, protein_out), dim=1)

        x = F.relu(self.fc1(concatenated))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()


class Transfer_RPI3(nn.Module):
    def __init__(self):
        super(Transfer_RPI3, self).__init__()

        self.rna_lstm = nn.LSTM(input_size=340, hidden_size=320, batch_first=True, bidirectional=True)
        self.rna_fc = nn.Linear(640, 1280)

        self.protein_lstm = nn.LSTM(input_size=1280, hidden_size=320, batch_first=True, bidirectional=True)
        self.protein_fc = nn.Linear(640, 1280)

        self.fc1 = nn.Linear(1280 * 2, 1280)
        self.fc2 = nn.Linear(1280, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, ran_kmer, protein_embedding):
        rna_embedding = ran_kmer.float().unsqueeze(1)
        rna_lstm_out, _ = self.rna_lstm(rna_embedding)
        rna_out = F.relu(self.rna_fc(rna_lstm_out[:, -1, :]))

        protein_embedding = protein_embedding.unsqueeze(1)
        protein_lstm_out, _ = self.protein_lstm(protein_embedding)
        protein_out = F.relu(self.protein_fc(protein_lstm_out[:, -1, :]))

        concatenated = torch.cat((rna_out, protein_out), dim=1)

        x = F.relu(self.fc1(concatenated))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()


class Transfer_RPI4(nn.Module):
    def __init__(self):
        super(Transfer_RPI4, self).__init__()

        self.rna_lstm = nn.LSTM(input_size=1280, hidden_size=320, batch_first=True, bidirectional=True)
        self.rna_fc = nn.Linear(640, 1280)

        self.protein_lstm = nn.LSTM(input_size=360, hidden_size=320, batch_first=True, bidirectional=True)
        self.protein_fc = nn.Linear(640, 1280)

        self.fc1 = nn.Linear(1280 * 2, 1280)
        self.fc2 = nn.Linear(1280, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, rna_embedding, protein_PAAC):
        rna_embedding = rna_embedding.unsqueeze(1)
        rna_lstm_out, _ = self.rna_lstm(rna_embedding)
        rna_out = F.relu(self.rna_fc(rna_lstm_out[:, -1, :]))

        protein_embedding = protein_PAAC.unsqueeze(1)
        protein_lstm_out, _ = self.protein_lstm(protein_embedding)
        protein_out = F.relu(self.protein_fc(protein_lstm_out[:, -1, :]))

        concatenated = torch.cat((rna_out, protein_out), dim=1)

        x = F.relu(self.fc1(concatenated))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()



class InteractionDataset(Dataset):
    def __init__(self, dataframe, rna_reps, protein_reps, rna_kmers, protein_PAACs):

        self.samples = []

        # 遍历DataFrame中的每一行，创建数据样本
        for _, row in dataframe.iterrows():
            rna_id = row['RNA_id']
            protein_id = row['protein_id']
            label = row['label']

            # 确保RNA和蛋白质的表示向量都在字典中
            if rna_id in rna_reps and protein_id in protein_reps:
                rna_vector = rna_reps[rna_id]
                protein_vector = protein_reps[protein_id]

                rna_kmer = rna_kmers[rna_id]
                protein_PAAC = protein_PAACs[protein_id]

                self.samples.append((rna_vector, protein_vector,  rna_kmer, protein_PAAC, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_traditional_representations(base_path, device):
    representations = {}
    for filename in os.listdir(base_path):
        if filename.endswith('.pt'):
            item_id = filename[:-3]
            file_path = os.path.join(base_path, filename)
            li = torch.load(file_path, map_location=device, weights_only=True)
            representations[item_id] = torch.tensor(li, device=device)
    return representations


def get_data(dataset_name, device):
    data = pd.read_csv(f'data/{dataset_name}/{dataset_name}.csv')
    rna_reps = load_pre_train_representations(f'data/{dataset_name}/rna_mean', device)
    protein_reps = load_pre_train_representations(f'data/{dataset_name}/protein_mean', device)
    rna_kmers = load_traditional_representations(f'data/{dataset_name}/rna_kmer', device)
    protein_PAACs = load_traditional_representations(f'data/{dataset_name}/protein_PAAC', device)
    dataset = InteractionDataset(data, rna_reps, protein_reps, rna_kmers, protein_PAACs)
    sample_num = len(dataset)
    print(f'Dataset "{dataset_name}" contains {sample_num} samples.')
    return dataset, sample_num


def train(dataset_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载数据
    dataset, sample_num = get_data(dataset_name, device)

    # 设置参数
    epochs, batch_size = get_hyperparameter(dataset_name)

    # 创建K折交叉验证的索引
    metrics_whole = {'Model': np.zeros(7)}
    K_FOLD = 5
    for fold in range(K_FOLD):
        print(f"Training fold {fold + 1}/{K_FOLD}")

        # 生成训练集和验证集的索引
        train_idx = [i for i in range(sample_num) if i % K_FOLD != fold]
        val_idx = [i for i in range(sample_num) if i % K_FOLD == fold]

        # 根据索引创建数据加载器
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=256, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = Transfer_RPI3().to(device)
        criterion = torch.nn.BCELoss()
        if dataset_name == 'RPI488':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # 训练和验证循环
        for epoch in range(epochs):
            model.train()
            for rna_emb, prot_emb, ran_kmer, protein_PAAC, labels in train_loader:
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(ran_kmer, prot_emb)
                outputs = outputs.float()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

        model.eval()
        label = []
        pred = []
        model_metrics = {'Model': np.zeros(7)}
        with torch.no_grad():
            for rna_emb, prot_emb, ran_kmer, protein_PAAC, labels in val_loader:
                labels = labels.to(device)
                outputs = model(ran_kmer, prot_emb)
                outputs = outputs.float()
                label.append(labels.float())
                pred.append(outputs)
        label = torch.cat(label, dim=0)
        pred = torch.cat(pred, dim=0)

        model_metrics['Model'] = np.array(calc_metrics(label.cpu(), pred.cpu()))
        print('ACC = ' + str(model_metrics['Model'][0]) + ' ' + 'SN = ' + str(
            model_metrics['Model'][1]) + ' '
              + 'SP = ' + str(model_metrics['Model'][2]) + ' ' + 'PRE = ' + str(
            model_metrics['Model'][3]) + ' '
              + 'F1_measure = ' + str(model_metrics['Model'][4]) + ' '
              + 'MCC = ' + str(model_metrics['Model'][5]) + ' ' + 'AUC = ' + str(
            model_metrics['Model'][6]))

        for key in model_metrics:
            metrics_whole[key] += model_metrics[key]

    for key in metrics_whole.keys():
        metrics_whole[key] /= K_FOLD
        print('\nMean metrics in {} fold:\n'.format(K_FOLD) + key + " : "
              + 'ACC = ' + str(metrics_whole[key][0]) + ' ' + 'SN = ' + str(metrics_whole[key][1]) + ' '
              + 'SP = ' + str(metrics_whole[key][2]) + ' ' + 'PRE = ' + str(metrics_whole[key][3]) + ' '
              + 'F1_measure = ' + str(metrics_whole[key][4]) + ' ' + 'MCC = ' + str(metrics_whole[key][5]) + ' '
              + 'AUC = ' + str(metrics_whole[key][6]))


if __name__ == "__main__":
    dataset_name = 'RPI2241'
    train(dataset_name)
