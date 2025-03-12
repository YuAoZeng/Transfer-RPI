import math
import pandas as pd
from sklearn.metrics import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from main import Transfer_RPI, InteractionDataset
from main import load_pre_train_representations


def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    if con_matrix.shape == (1, 1):
        TP = float(con_matrix[0][0])
        FP = 0.0
        FN = 0.0
        TN = 0.0
    else:
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
    true_num = TN + TP
    print('number of correct predictions', true_num)
    return Acc, Sn, Sp, Pre, F1_measure, MCC, AUC


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = pd.read_csv('data/RPI2241/RPI2241.csv')
    rna_reps = load_pre_train_representations('data/RPI2241/rna_mean', device)
    protein_reps = load_pre_train_representations('data/RPI2241/protein_mean', device)
    data['label'] = data['label'].astype(float)

    dataset = InteractionDataset(data, rna_reps, protein_reps)
    dataset.report_missing_ids()
    sample_num = len(dataset)
    print(f'Dataset RPI2241 contains {sample_num} samples.')

    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = Transfer_RPI().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 训练和验证循环
    for epoch in range(200):
        model.train()
        for rna_emb, prot_emb, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(rna_emb, prot_emb)
            outputs = outputs.float()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
    # 保存模型权重
    # torch.save(model.state_dict(), 'data/model_weights/RPI2241_train_weight.pth')

def val():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载数据
    data = pd.read_csv('data/NPInter/classification/total_pairs.csv')
    rna_reps = load_pre_train_representations('data/NPInter/rna_mean', device)
    protein_reps = load_pre_train_representations('data/NPInter/protein_mean', device)

    data['label'] = data['label'].astype(float)


    dataset = InteractionDataset(data, rna_reps, protein_reps)
    dataset.report_missing_ids()
    sample_num = len(dataset)
    print(sample_num)

    val_loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model = Transfer_RPI().to(device)
    model.load_state_dict(torch.load('data/model_weights/RPI2241_train_weight.pth'))

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
    print('ACC = ' + str(model_metrics['Model'][0]))

if __name__ == "__main__":
    val()
