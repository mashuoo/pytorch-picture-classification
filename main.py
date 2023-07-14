import torch
import torch.nn as nn
import torch.optim as optim

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # 输入层到隐藏层
        self.fc2 = nn.Linear(64, 32)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(32, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用Sigmoid激活函数
        return x

# 创建模型实例
model = NeuralNetwork()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

# 准备训练数据和标签
train_data = torch.randn(100, 10)
train_labels = torch.randint(0, 2, (100, 1)).float()

# 开始训练
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清除梯度
    outputs = model(train_data)  # 前向传播
    loss = criterion(outputs, train_labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 使用模型进行预测
test_data = torch.randn(10, 10)
predictions = model(test_data)
print("预测结果:")
for i in range(len(predictions)):
    print(f"样本 {i+1}: {predictions[i].item()}")
