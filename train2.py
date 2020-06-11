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
net = net.to(device)

# 定义优化器和损失函数
# SGD 就是随机梯度下降
opt_SGD = torch.optim.SGD(net.parameters(), lr=args.lr)
# momentum 动量加速,在SGD函数里指定momentum的值即可
opt_Momentum = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.8)
# RMSprop 指定参数alpha
opt_RMSprop = torch.optim.RMSprop(net.parameters())
# Adam 参数betas=(0.9, 0.99)
opt_Adam = torch.optim.Adam(net.parameters())
optimizer = opt_Adam

# Loss函数
loss_function = torch.nn.MSELoss() # 最小均方误差
# 神经网络训练过程
plt.ion() # 动态学习过程展示
# plt.xlim(0, 100)
# plt.ylim(0,100)

dataLoader = data_loader.get_loader(args.batch_size)
for epoch in range(args.num_epochs):
  for i, (x, y) in enumerate(dataLoader):
    x = x.to(device)
    y = y.to(device)
    prediction = net(x) # 把数据x喂给net，输出预测值
    loss = loss_function(prediction, y) # 计算两者的误差，要注意两个参数的顺序
    optimizer.zero_grad() # 清空上一步的更新参数值
    loss.backward() # 误差反相传播，计算新的更新参数值
    optimizer.step() # 将计算得到的更新值赋给net.parameters()

    # Plot the graph
    plt.scatter(y.data.cpu().numpy(), prediction.cpu().data.numpy(), label='Fitted line')
    plt.pause(0.5)

  # if (epoch + 1) % 5 == 0:
  print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, loss.item()))

  # Save the model checkpoint
  if (epoch + 1) % args.save_epoch == 0:
   torch.save(net.state_dict(), './ckpt/model{}.ckpt'.format(epoch + 1))

plt.ioff()
plt.show()