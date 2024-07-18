import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder

from torchvision.models import resnet


class MyTransformerModel(nn.Module):
    def __init__(self):
        super(MyTransformerModel, self).__init__()
        self.transformer = TransformerWrapper(
            num_tokens=1000,
            max_seq_len=100,
            attn_layers=Decoder(
                dim=512,
                depth=6,
                heads=8,
                attn_flash=True
            )
        )

        # 添加一个全连接层用于二分类
        self.fc_binary = nn.Linear(512, 1)  # 输入维度为 512，输出维度为 1

        # 添加 sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        transformer_output = self.transformer(x)
        # print(x.shape)
        # 新的全连接层
        binary_output = self.fc_binary(transformer_output)

        # 应用 sigmoid 激活函数
        binary_output = self.sigmoid(binary_output)

        return binary_output


# 创建模型实例
model = MyTransformerModel()

# 将模型移动到设备（CPU 或 GPU）
device = torch.device("cpu")
model.to(device)
# # 打印模型结构
print(model)
