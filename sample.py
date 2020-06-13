import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import model
import config
import data_loader


def read_all_model_file(path):
    files = []
    if os.path.isfile(path): return [path]
    path_list = os.listdir(path)
    for file in path_list:
        file = os.path.join(path,file)
        if os.path.isdir(file):
            files += read_all_model_file(file)
        else:
            files.append(file)
    return files

def predict_props(path,set='val'):
    model_list = read_all_model_file(path)
    plot_list = []
    for i,pth in enumerate(model_list):
        model = torch.load(pth, map_location=torch.device('cpu'))
        model.eval() 
        compose,prop = data_loader.data_read(set)
        out = model(compose)
        loss = torch.nn.MSELoss() (prop, out)
        prop = prop.data.cpu().numpy()
        out = out.data.cpu().numpy()
        R2 = 1-loss/np.var(prop)
        rmse = np.sqrt(loss.detach().numpy())
        print('loss:{:.4f} R2:{} File:{}'.format(loss,R2,pth))
        plot_list.append([prop, out, config.color[i], 'RMSE:{:.4f}'.format(rmse)])
    plt.ion()
    plt.plot([20,120],[20,120])
    for param in plot_list: 
        plt.scatter(param[0], param[1], color=param[2], label=param[3])
        plt.pause(0.5)
    plt.legend()
    plt.ioff()
    plt.show()


def predict_design(path,set='val'):
    model_list = read_all_model_file(path)
    plot_list = []
    for i,pth in enumerate(model_list):
        model = torch.load(pth, map_location=torch.device('cpu'))
        model.eval() 
        compose,prop = data_loader.data_read(set)
        out = model(prop)
        loss = torch.nn.MSELoss() (compose, out)
        prop = prop.data.cpu().numpy()
        compose = compose.data.cpu().numpy()
        out = out.data.cpu().numpy()
        R2 = 1-loss/np.var(compose)
        rmse = np.sqrt(loss.detach().numpy())
        print('loss:{:.4f} R2:{} File:{}'.format(loss,R2,pth))
        plot_list.append([compose, out, config.color[i], 'RMSE:{:.4f}'.format(rmse)])
    plt.ion()
    plt.plot([20,120],[20,120])
    for param in plot_list: 
        plt.scatter(param[0], param[1], color=param[2], label=param[3])
        plt.pause(0.5)
    plt.legend()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    prediction_type = 2 # 1:性能预测，2:组成预测
    pth_path = 'pth'
    if prediction_type==1:
        predict_props('pkl/props')
    elif prediction_type==2:
        predict_design('pkl/design')