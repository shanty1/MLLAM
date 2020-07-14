import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 定义一个构建神经网络的类 
class SingleNet(nn.Module):  # 继承torch.nn.Module类
    def __init__(self, n_input, n_hidden, n_output):
        super(SingleNet, self).__init__()  # 获得Net类的超类（父类）的构造方法
        # 定义神经网络的每层结构形式
        # 各个层的信息都是Net类对象的属性
        self.hidden = nn.Linear(n_input, n_hidden)  # 隐藏层线性输出
        self.predict = nn.Linear(n_hidden, n_output)  # 输出层线性输出

    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self, input_x):
        out = F.relu(self.hidden(input_x))  # 对隐藏层的输出进行relu激活
        out = self.predict(out)
        return out


# 定义一个构建神经网络的类
class MultipleNet(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output):
        super(MultipleNet, self).__init__()
        self.hidden = self._make_layers(n_input, n_hiddens)
        self.predict = nn.Linear(n_hiddens[-1], n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = self.predict(out)
        return out

    def _make_layers(self, in_dim, n_hiddens):
        layers = []
        for x in n_hiddens:
            layers += [nn.Linear(in_dim, x),
                       nn.BatchNorm1d(x),
                       nn.Dropout(0.1),
                       nn.ReLU(inplace=True)]
            in_dim = x
        return nn.Sequential(*layers)
