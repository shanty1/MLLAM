import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import model
import config
import data_loader


# 参数配置
args = config.args
device = config.device

pth_name='best/model_best.pkl'
net=torch.load('./pth/'+pth_name,
                               map_location=torch.device('cpu'))
net = net.to(device)
net.eval() 
 
plt.ion() # 动态学习过程展示

x,y = data_loader.data_read(name="val")
x = x.to(device)
y = y.to(device)

prediction = net(x) # 把数据x喂给net，输出预测值
loss = torch.nn.MSELoss() (prediction, y) # 计算两者的误差，要注意两个参数的顺序

print(' Loss: {:.4f}'.format(loss.item()))
# Plot the graph
x = y.data.cpu().numpy()
y = prediction.cpu().data.numpy()
plt.plot(20,120)
plt.scatter(x, y, color=config.color[3])
plt.pause(0.5)
plt.ioff()


plt.show()

file_list = os.listdir('./pth')
for file in file_list:
    if os._isdir