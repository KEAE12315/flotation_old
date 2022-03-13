import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from preprocess import Normalize, create_data


class MLP(torch.nn.Module):
    def __init__(self, x_n, y_n):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(x_n, 100)
        self.fc2 = torch.nn.Linear(100, 20)
        self.fc3 = torch.nn.Linear(20, y_n)

    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        dout = F.elu(self.fc3(dout))
        return dout


def train(n_epochs):
    lossfunc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(data)    # 得到预测值
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))


def test():
    lossfunc = torch.nn.MSELoss()
    test_loss = 0.0

    norm = Normalize()
    with torch.no_grad():  # 训练集中不需要反向传播
        for x, y in test_loader:
            # x = norm.code(x)
            outputs = model(x)
            loss = lossfunc(outputs, y)
            test_loss += loss.item()*x.size(0)

            outputs = norm.decode(outputs.tolist()[0])
            y = norm.decode(y.tolist()[0])
            print(y, outputs)

        # test_loss = test_loss / len(test_loader.dataset)
        # print('Testing Loss: {:.6f}'.format(test_loss))


if __name__ == '__main__':
    df = pd.read_csv('dataset/feed.csv', index_col=0)
    train_data, test_data, x_n, y_n = create_data(df[:], split=0.9, ls=20)
    train_loader = DataLoader(train_data, batch_size=7, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = MLP(x_n, y_n)


    for x, y in train_data:
        print(x.size(), y)
        break

    train(100)
    test()

    model_path = 'model/mlp.pt'
    torch.save(model.state_dict(), model_path)

    # model = MLP()
    # model.load_state_dict(torch.load(model_path))

    # model.eval()