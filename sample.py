import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import model
import config
import data_loader

# 参数配置
args = config.args
device = config.device

# 定义神经网络
simple_net = model.SingleNet(args.n_input, args.n_hidden, args.n_output)
multipleNet = model.MultipleNet(args.n_input, args.n_hiddens, args.n_output)
net = multipleNet

net.load_state_dict(torch.load('./model999000.ckpt',
                               map_location=torch.device('cpu')))
net = net.to(device)
 
# Loss函数
loss_function = torch.nn.MSELoss() # 最小均方误差
# 神经网络训练过程
plt.ion() # 动态学习过程展示
# plt.xlim(0, 100)
# plt.ylim(0,100)

x,y = data_loader.data_read(name="test")
x = x.to(device)
y = y.to(device)
prediction = net(x) # 把数据x喂给net，输出预测值
loss = loss_function(prediction, y) # 计算两者的误差，要注意两个参数的顺序

print(' Loss: {:.4f}'.format(loss.item()))
# Plot the graph
plt.scatter(y.data.cpu().numpy(), prediction.cpu().data.numpy(), label='Fitted line')
plt.plot([0,100],[0,100])
plt.pause(0.5)

plt.ioff()
plt.show()
