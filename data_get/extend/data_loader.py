import torch

data_dir = "./data/"
# 输入字段/特征
x_names = ['']
# 输出字段/标签
y_names = ['']

# HRB,Mn,Mo,C,Al,N,Cr,P,Cu,S,Si,Ni
def data_read(name="train"):
    list_x = []
    list_y = []
    file = open(data_dir+name+".txt")
    for line in file:
        if not line.strip(): continue
        a = [float(x) for x in line.replace("\n","").split(",")]
        list_x.append(a[1:])
        list_y.append(a[:1])
    file.close()
    list_x = torch.tensor(list_x)
    list_y = torch.tensor(list_y)
    return list_x,list_y


def get_loader(batch_size,name="train",desc=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if desc:
        y,x = data_read(name)
    else:
        x,y = data_read(name)
    torch_dataset = torch.utils.data.TensorDataset(x,y)
    data_loader = torch.utils.data.DataLoader(
        dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return data_loader



def get_dataloaders_train_val(file_name, batch_size_for_train, batch_size_for_val, desc=False):
    return {
        'train': get_loader(batch_size_for_train, file_name, desc),
        # 无测试数据，暂时用训练集代替
        'val': get_loader(batch_size_for_val, file_name, desc)
    }
