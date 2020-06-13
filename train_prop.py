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


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs, save_epoch,save_name='model'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(1,num_epochs+1):
        first = True # 标记当前是一个epoch
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: #(训练一次测试一下)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            data_size = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                data_size += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)  # 本次Iterate*样本数=本次的总样本loss（防止最后一个batch大小不同，或train与val的不同）
            if phase == 'train':
                scheduler.step()
  
            epoch_loss = running_loss / data_size   # 一个epoch的平均loss

            # show each epoch
            if args.show_each_epoch:
                if first:
                    print('\nEpoch {}/{}\n{}'.format(epoch, num_epochs, '-' * 10))
                    first = False
                print('{:5s} Loss: {:.4f} LR: {:.4f} data_size：{}'.format(
                    phase, epoch_loss, optimizer.param_groups[0]['lr'], data_size))  # 一个epoch更新
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_self, os.path.join(os.path.dirname(__file__), 'pth/best/{}{}.pkl'.format(save_name，epoch)))
            
            if (epoch) % save_epoch == 0:
                # 保存模型，此处判断保证保存一次，并且展示的loss是val的
                if phase == 'val':
                    torch.save( model.state_dict(), './pth/{}{}-valLoss-{}.pth'.format(epoch,save_name, epoch_loss))
                if not args.show_each_epoch:
                    if first:
                        print('\nEpoch {}/{}\n{}'.format(epoch, num_epochs, '-' * 10))
                        first = False
                    print('{:5s} Loss: {:.4f} LR: {:.4f} data_size：{}'.format(
                        phase, epoch_loss, optimizer.param_groups[0]['lr'], data_size))
            # 每次epoch下的train/val结束

        # 每次epoch结束
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
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
if __name__ == "__main__":
    plt.ion()   # interactive mode
    dataloaders = data_loader.get_dataloaders_train_val(
        args.batch_size_train, args.batch_size_val)
    criterion = torch.nn.MSELoss()
    # exec train

    model_self = model.MultipleNet(args.n_input, args.n_hiddens, args.n_output).to(device)
    opt = torch.optim.SGD(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(copy.deepcopy(dataloaders), model_self, criterion, opt,
                            scheduler, args.num_epochs, args.save_epoch,'model1')

    model_self = model.MultipleNet(args.n_input, args.n_hiddens, args.n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    train_model(copy.deepcopy(dataloaders), model_self, criterion, opt,
                            scheduler, args.num_epochs, args.save_epoch,'model2') 

    model_self = model.MultipleNet(args.n_input, args.n_hiddens, args.n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(opt, step_size=STEPLR_STEP, gamma=0.1)
    train_model(copy.deepcopy(dataloaders), model_self, criterion, opt,
                            scheduler, args.num_epochs, args.save_epoch,'model3') 

    model_self = model.MultipleNet(args.n_input, [100, 200, 50], args.n_output).to(device)
    opt = torch.optim.Adam(model_self.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(opt, step_size=STEPLR_STEP, gamma=0.1)
    train_model(copy.deepcopy(dataloaders), model_self, criterion, opt,
                            scheduler, args.num_epochs, args.save_epoch,'model4') 
    
    ######################################################################













    
    # plt.ioff()
    # plt.show()
