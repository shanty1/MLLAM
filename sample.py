import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import model
import config
import data_loader
from sklearn.metrics import r2_score

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

R2s=[]
def predict_props(path,set='val'):
    model_list = read_all_model_file(path)
    plot_list = []
    x_compose,y_prop = data_loader.data_read(set)
    prop = y_prop.data.cpu().numpy()
    criterion = torch.nn.MSELoss()
    for i,pth in enumerate(model_list):
        model = torch.load(pth, map_location=torch.device('cpu'))
        model.eval() 
        out = model(x_compose)
        loss = criterion(y_prop, out)
        out = out.data.cpu().numpy()
        # R2 = 1-loss/np.var(prop)
        R2 = r2_score(prop,out)
        R2s.append(R2)
        rmse = np.sqrt(loss.detach().numpy())
        print('loss:{:.4f} R2:{} File:{}'.format(loss,R2,pth))
        plot_list.append([prop, out, config.randomcolor(), 'RMSE:{:.4f},r2:{:.4f}'.format(rmse,R2)])
    plt.ion()
    for param in plot_list: 
        plt.scatter(param[0], param[1], color=param[2], label=param[3])
        plt.pause(0.5)
    plt.legend()
    plt.ioff()
    plt.show()


def predict_design(path,set='val'):
    model_list = read_all_model_file(path)
    plot_list = []
    y_compose,x_prop = data_loader.data_read(set)
    criterion = torch.nn.MSELoss()
    for i,pth in enumerate(model_list):
        model = torch.load(pth, map_location=torch.device('cpu'))
        model.eval() 
        out = model(x_prop)
        loss = criterion(y_compose, out)
        compose = y_compose.data.cpu().numpy()
        out = out.data.cpu().numpy()
        # R2 = 1-loss/np.var(compose)
        R2 = r2_score(compose, out)
        rmse = np.sqrt(loss.detach().numpy())
        print('loss:{:.4f} R2:{} File:{}'.format(loss,R2,pth))
        plot_list.append([compose, out, config.randomcolor(), 'RMSE:{:.4f},R2:{:.4F}'.format(rmse,R2)])
    plt.ion()
    for param in plot_list: 
        plt.scatter(param[0], param[1], color=param[2], label=param[3])
        plt.pause(0.5)
    plt.legend()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    prediction_type = 1 # 1:性能预测，2:组成预测
    if prediction_type==1:
        predict_props('pkl/props')
        print('bestR2:{}'.format(max(R2s)))
    elif prediction_type==2:
        predict_design('pkl/design')
