import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from rna_model import MyTransformerModel
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

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


#def train_model(model, fold, train_loader, val_loader, criterion, optimizer, device, num_epochs=30):
def train_model(model, fold, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    # model：要训练的模型。fold：交叉验证的折数。train_loader：训练数据的数据加载器。val_loader：验证数据的数据加载器。criterion：损失函数。optimizer：优化器。device：设备（例如 CPU 或 GPU）。num_epochs：训练的总轮数，默认为 30。
    #model.to(device)
    best_accuracy = 0.0
    for epoch in range(num_epochs):  # 30轮
        total_loss = 0.1  # 初始化一些变量，用于计算训练损失和准确率。
        total_correct = 0.2
        total_samples = 0.1
        model.train()

        for inputs, targets in train_loader:
            # inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        train_accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # 在验证集上评估模型
        val_accuracy, val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs},Training Accuracy: {train_accuracy:.4f}, Loss: {total_loss:.4f}\
              , Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # 保存效果最好的模型
            torch.save(model.state_dict(), fr"C:/Users/lenovo/Desktop/model/rna_model/best_model/best_model_fold_{fold + 1}.pt")


#def evaluate_model(model, val_loader, criterion, device):
def evaluate_model(model, val_loader, criterion):
    # 评估逻辑
    #model.to(device)
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            #inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    return val_accuracy, val_loss

def train_fold(n_fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gene_data = torch.load(r'C:\Users\lenovo\Desktop\model\rna_model\gene_train.pth')

    num_splits = n_fold
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(gene_data)):
        train_data = [gene_data[i] for i in train_indices]
        val_data = [gene_data[i] for i in val_indices]

        train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=100, shuffle=False)

        model = MyTransformerModel()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        #train_model(model, fold, train_loader, val_loader, criterion, optimizer, device)
        train_model(model, fold, train_loader, val_loader, criterion, optimizer)



def main():
    train_fold(5)


if __name__ == '__main__':
    main()
