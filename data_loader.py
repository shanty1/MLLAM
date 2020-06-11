import torch

data_dir = "./data/"

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


def get_loader(batch_size,name="train"):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    torch_dataset = torch.utils.data.TensorDataset(*data_read(name))
    data_loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def load_train_data(batch_size):
    return get_loader(batch_size,data_read("train"))
def load_test_data(batch_size):
    return get_loader(batch_size,data_read("test"))
