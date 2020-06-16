import torch
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import model
import config
import data_loader
import copy
import os
import time
import _thread

# 参数配置
args = config.args
device = config.device


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs, save_epoch,save_name='model',save_path='./pkl'):
    isReduceLROnPlateau = False
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        isReduceLROnPlateau = True
    since = time.time()

    best_model_wts = None
    best_loss = float("inf")

    trainLoss = []
    valLoss = []
    lrs = []
    epochs = []
    plt.ion()
    for epoch in range(1,num_epochs+1):
        epochs += [epoch]
        lrs += [optimizer.param_groups[0]['lr']]

        # train:
        model.train()
        running_loss = 0.0
        data_size = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # statistics
            data_size += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)  # 本次Iterate*样本数=本次的总样本loss（防止最后一个batch大小不同，或train与val的不同）
        
        epoch_loss = running_loss / data_size   # 一个epoch的平均loss
        trainLoss += [epoch_loss]
        
        # validation:
        model.eval()
        running_loss = 0.0
        data_size = 0
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            torch.set_grad_enabled(False)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # statistics
            data_size += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)  # 本次Iterate*样本数=本次的总样本loss（防止最后一个batch大小不同，或train与val的不同）
        
        epoch_loss = running_loss / data_size   # 一个epoch的平均loss
        valLoss += [epoch_loss]

        # auto update lr
        if scheduler:
            if isReduceLROnPlateau:
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
        
        # show each epoch
        if args.show_each_epoch:
            print('Epoch {}/{}\n{}'.format(epoch, num_epochs, '-' * 10))
            print('train_loss: {:.4f}\n  val_loss: {:.4f}\nlearning_rate: {:.4f}\n'.format(
                trainLoss[-1], valLoss[-1], optimizer.param_groups[0]['lr']))  # 一个epoch更新

        # deep copy the model(lav loss)
        if valLoss[-1] < best_loss:
            best_loss = valLoss[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, '{}/{}_{}-trainLoss_{:.4f}-valLoss_{:.4f}.pkl'.format(
                save_path, save_name, epoch, trainLoss[-1], valLoss[-1]))

    # printHistory(epochs,trainLoss,valLoss,lrs)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, '{}/best/{}.pkl'.format(save_path, save_name))
    return model


# 网络
net_simple = model.SingleNet(args.n_input, args.n_hidden, args.n_output)
net_multiple = model.MultipleNet(args.n_input, args.n_hiddens, args.n_output)
# 优化器
# opt_SGD = torch.optim.SGD(net.parameters(), lr=args.lr)  # SGD 就是随机梯度下降
# opt_Momentum = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.8) # momentum 动量加速,在SGD函数里指定momentum的值即可
# opt_RMSprop = torch.optim.RMSprop(net.parameters())  # RMSprop 指定参数alpha
# opt_Adam = torch.optim.Adam(net.parameters(), lr=args.lr) # Adam 参数betas=(0.9, 0.99)
# 学习率调度器
# cosineAnnealingLR_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)  # lr随着epoch的变化图类似于cos
# stepLR_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEPLR_STEP, gamma=0.1) # new_lr = initial_lr * γ^(epoch//step_size)
# 损失函数
loss_function_mse = torch.nn.MSELoss()  # 最小均方误差

#cos优化器T_max Maximum number of iterations
T_MAX = args.save_epoch
STEPLR_STEP = args.save_epoch
T_MAX = 200
STEPLR_STEP = 200
if __name__ == "__main__":

    dataloaders = data_loader.get_dataloaders_train_val(args.batch_size_train, args.batch_size_val)
    criterion = torch.nn.MSELoss()

    # exec train props
    n_input,n_output = args.n_input,args.n_output

    model_self = model.MultipleNet(n_input,   [500,200], n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    train_model(dataloaders, model_self, criterion, opt,
                None, args.num_epochs, args.save_epoch, 'model1', './pkl/props/')

    model_self = model.MultipleNet(n_input,   [500,200], n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model2', './pkl/props/')
                            
    model_self = model.MultipleNet(n_input,  [500,200], n_output).to(device)
    opt = torch.optim.SGD(model_self.parameters(), lr=args.lr, momentum=0.8)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model3', './pkl/props/')
    
    model_self = model.MultipleNet(n_input,  [200,500,200], n_output).to(device)
    opt = torch.optim.SGD(model_self.parameters(), lr=args.lr, momentum=0.8)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model4', './pkl/props/')
    
    model_self = model.MultipleNet(n_input,  [200,500,200], n_output).to(device)
    opt = torch.optim.SGD(model_self.parameters(), lr=args.lr, momentum=0.8)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model5', './pkl/props/')
   
    model_self = model.MultipleNet(n_input,  [200,500,200], n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model5', './pkl/props/')

    ######################################################################
    # exec train design
    n_input, n_output = args.n_output, args.n_input

    dataloaders = data_loader.get_dataloaders_train_val(args.batch_size_train, args.batch_size_val,True)

    model_self = model.MultipleNet(n_input, [200, 500, 200,100], n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(copy.deepcopy(dataloaders), model_self, criterion, opt,
                            scheduler, args.num_epochs, args.save_epoch,'model1','./pkl/design/') 

    model_self = model.MultipleNet(n_input,   [500,200], n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model2', './pkl/design/')
                            
    model_self = model.MultipleNet(n_input,  [500,200], n_output).to(device)
    opt = torch.optim.SGD(model_self.parameters(), lr=args.lr, momentum=0.8)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model3', './pkl/design/')
    
    model_self = model.MultipleNet(n_input,  [200,500,200], n_output).to(device)
    opt = torch.optim.SGD(model_self.parameters(), lr=args.lr, momentum=0.8)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model4', './pkl/design/')
    
    model_self = model.MultipleNet(n_input,  [200,500,200], n_output).to(device)
    opt = torch.optim.SGD(model_self.parameters(), lr=args.lr, momentum=0.8)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model5', './pkl/design/')
   
    model_self = model.MultipleNet(n_input,  [200,500,200], n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt)
    train_model(dataloaders, model_self, criterion, opt,
                scheduler, args.num_epochs, args.save_epoch, 'model5', './pkl/design/')




