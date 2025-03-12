import torch
import math
import os
import pandas as pd
from sklearn.metrics import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset


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


class InteractionDataset(Dataset):
    def __init__(self, dataframe, rna_reps, protein_reps):
        """
        初始化数据集
        :param dataframe: 包含列 'RNA_id' (RNA ID), 'protein_id' (蛋白质 ID) 和 'Label' 的pandas DataFrame。
        :param rna_reps: 一个字典，键为RNA ID，值为RNA的表示向量。
        :param protein_reps: 一个字典，键为蛋白质 ID，值为蛋白质的表示向量。
        """
        self.samples = []
        self.missing_rna_ids = set()  # 存储缺失的RNA ID
        self.missing_protein_ids = set()  # 存储缺失的蛋白质ID

        # 遍历DataFrame中的每一行，创建数据样本
        for _, row in dataframe.iterrows():
            rna_id = row['RNA_id']
            protein_id = row['protein_id']
            label = row['label']

            # 检查RNA和蛋白质的ID是否在字典中
            if rna_id not in rna_reps:
                self.missing_rna_ids.add(rna_id)
            if protein_id not in protein_reps:
                self.missing_protein_ids.add(protein_id)

            if rna_id in rna_reps and protein_id in protein_reps:
                rna_vector = rna_reps[rna_id]
                protein_vector = protein_reps[protein_id]

                self.samples.append((rna_vector, protein_vector, label))

    def report_missing_ids(self):
        # 报告缺失的ID
        if self.missing_rna_ids:
            print("缺失的RNA ID:", self.missing_rna_ids)
        if self.missing_protein_ids:
            print("缺失的蛋白质ID:", self.missing_protein_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = (TP) / (TP + FP) if (TP+FP) > 0 else 0
    F1_measure = (2*Sn*Pre)/(Sn+Pre)
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return Acc, Sn, Sp, Pre, F1_measure, MCC, AUC


def load_pre_train_representations(base_path, device):
    representations = {}
    for filename in os.listdir(base_path):
        if filename.endswith('.pt'):
            # 假设文件名格式为 'id_mean.pt'，从文件名中提取ID
            item_id = filename[:-8]  # 删除末尾的 '_mean.pt' 获取ID
            file_path = os.path.join(base_path, filename)
            representations[item_id] = torch.load(file_path, map_location=device, weights_only=True)
    return representations


def get_data(dataset_name, device):
    data = pd.read_csv(f'data/{dataset_name}/{dataset_name}.csv')
    rna_reps = load_pre_train_representations(f'data/{dataset_name}/rna_mean', device)
    protein_reps = load_pre_train_representations(f'data/{dataset_name}/protein_mean', device)
    data['label'] = data['label'].astype(float)
    dataset = InteractionDataset(data, rna_reps, protein_reps)
    dataset.report_missing_ids()
    sample_num = len(dataset)
    print(f'Dataset "{dataset_name}" contains {sample_num} samples.')
    return dataset, sample_num

def get_hyperparameter(dataset_name):
    if dataset_name == "RPI369":
        epochs = 200
        batch_size = 32
    elif dataset_name == "RPI488":
        epochs = 200
        batch_size = 32
    elif dataset_name == "RPI1807":
        epochs = 100
        batch_size = 128
    elif dataset_name == "RPI2241":
        epochs = 200
        batch_size = 128
    elif dataset_name == "NPInter":
        epochs = 300
        batch_size = 512
    else:
        epochs = 200  # 默认值
        batch_size = 128  # 默认值

    return epochs, batch_size



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
        model = Transfer_RPI().to(device)
        criterion = torch.nn.BCELoss()
        if dataset_name == 'RPI488':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # 训练和验证循环
        for epoch in range(epochs):
            model.train()
            for rna_emb, prot_emb, labels in train_loader:
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(rna_emb, prot_emb)
                outputs = outputs.float()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

        model.eval()
        label = []
        pred = []
        model_metrics = {'Model': np.zeros(7)}
        with torch.no_grad():
            for rna_emb, prot_emb, labels in val_loader:
                labels = labels.to(device)
                outputs = model(rna_emb, prot_emb)
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
    dataset_name = 'NPInter'
    train(dataset_name)