import sys
import torch
from torch.utils.data import DataLoader
from rna_model import MyTransformerModel
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch.nn as nn
import matplotlib.pyplot as plt


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



def plot_roc_curve(fpr, tpr, roc_auc):
    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 保存ROC曲线图像
    plt.savefig('dna_model_1/roc_curve.png')


def plot_pr_curve(precision, recall, pr_auc):
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    # 保存PR曲线图像
    plt.savefig('dna_model_1/pr_curve.png')


def load_samples_with_labels(file_path):
    samples_with_labels = []
    current_label = None  # 当前标签初始化为None

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
                # 这里假设没有标记的行都是序列行
                sequence = line
                if current_label is not None:
                    # 只有在设置了当前标签后才添加序列
                    samples_with_labels.append({'sequence': sequence, 'label': current_label})
                else:
                    # 如果在序列之前没有遇到标记，则打印错误信息
                    print(f"无法识别的行，缺少标签: {line}")

    return samples_with_labels


def evaluate_model(model, val_loader, criterion, device):
    # 评估逻辑
    model.to(device)
    model.eval()
    all_labels = []
    all_values = []
    all_predictions = []
    val_loss = 0.0
    val_accuracy = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, outputs = model(inputs)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_values.extend(logits.cpu()[:, 1].numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    fpr, tpr, thresholds = roc_curve(all_labels, all_values)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(all_labels, all_values)
    pr_auc = auc(recall, precision)
    val_accuracy = correct / total
    return val_accuracy, val_loss, conf_matrix, fpr, tpr, roc_auc, precision, recall, pr_auc


def test(test_file, model_path, result_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载测试数据
    test_data = load_samples_with_labels(test_file)

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # 创建模型
    #input_channels = 1
    #conv_out_channels = 16
    #kernel_size = 3
    #num_classes = 2
    #model = GeneAnalysisModel(input_channels, conv_out_channels, kernel_size, num_classes).to(device)
    model = MyTransformerModel()
    # 加载之前训练好的模型参数
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 在测试集上评估模型
    test_accuracy, test_loss, conf_matrix, fpr, tpr, roc_auc, precision, recall, pr_auc = evaluate_model(model,
                                                                                                         test_loader,
                                                                                                         criterion,
                                                                                                         device)
    torch.save({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'precision': precision, 'recall': recall, 'pr_auc': pr_auc}
               , result_path)

    # 计算指标
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    F1 = 2 * TP / (2 * TP + FP + FN)

    print("Sensitivity (SN):", SN)
    print("Specificity (SP):", SP)
    print("Accuracy (ACC):", ACC)
    print("Matthews Correlation Coefficient (MCC):", MCC)
    print("F1 Score (F1):", F1)
    print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")
    # 画 ROC 曲线
    plot_roc_curve(fpr, tpr, roc_auc)
    # 画 PR 曲线
    plot_pr_curve(precision, recall, pr_auc)


def main():
    test_file = r'C:\Users\lenovo\Desktop\model\iRNA-m6A\iRNA-m6A\independent\independent\h_b_Test.fa'
    model_path = 'dna_model_1/best_model/best_model.pt'  # 换成你训练好的模型的路径#******************************
    result_path = r'C:/Users/lenovo/Desktop/model/rna_model/model_result.pt'
    test(test_file, model_path, result_path)


if __name__ == '__main__':
    main()















