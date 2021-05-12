
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class Rehearsal_MNIST_CIFAR_SVNH(Dataset):
    def __init__(self,data_type_list=[], file_path="data/", num_per_class=50,
                 train=True, transform_list=[], label_align=True):
        self.transform_list = transform_list
        self.label_align = label_align


        self.num_per_class = num_per_class
        self.data_list = self.produce_rehearsal(data_type_list=data_type_list,file_path=file_path, train=train)


    def __len__(self):
        lens = 0
        for data in self.data_list:
            lens += len(data['x'])
        return lens

    def __getitem__(self, item):
        index = int(item / int(self.num_per_class*10))

        item = int(item - index*int(self.num_per_class*10))

        data = self.data_list[index]
        x, y = data['x'][item], data['y'][item]
        y = y + index*10
        # x, y = data['x'][item], index
        x = Image.fromarray(x)
        if self.transform_list is not None:
            x = self.transform_list[index](x)
        return x, y

    def produce_rehearsal(self,data_type_list=[], file_path="",train=True):

        data_list=[]
        for i in range(len(data_type_list)):
            data_type = data_type_list[i]
            if data_type is "cifar10":
                train_data = torchvision.datasets.CIFAR10(
                    root=file_path,
                    train=train,
                    download=True,
                )
                x_train = train_data.data
                y_train = train_data.targets
                y_train = np.array(y_train)

            elif data_type is "SVNH":
                if train:
                    mode = "train"
                else:
                    mode = "test"
                train_data = torchvision.datasets.SVHN(
                    root=file_path,
                    split=mode,
                    download=True,
                )
                x_train = train_data.data.transpose(0, 2, 3, 1)
                y_train = train_data.labels

            elif data_type is "MNIST":

                train_data = torchvision.datasets.MNIST(
                    root=file_path,
                    train=train,
                    download=True,
                )
                x_train = train_data.data.numpy()
                # x = Image.fromarray(train_data.data.numpy(), mode='L')
                # x_train = np.concatenate([x, x, x], axis=1)
                # x_train = torchvision.transforms.Resize(32)(x_train)


                y_train = train_data.targets

            rehearsal_data = None
            rehearsal_label = None
            for i in range(1, 10+1):
                a1 = y_train >= i - 1
                a2 = y_train < i
                index = a1 & a2
                task_train_x = x_train[index]
                label = y_train[index]
                index = np.random.choice(task_train_x.shape[0], self.num_per_class)
                tem_data = task_train_x[index]
                tem_label = label[index]
                if rehearsal_data is None:
                    rehearsal_data = tem_data
                    rehearsal_label = tem_label
                else:
                    rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                    rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

            data_list.append(
                {
            "x": rehearsal_data,
            "y": rehearsal_label,}
            )

        return data_list

class to3channels(object):


    def __call__(self, tensor):

        return torch.cat([tensor, tensor, tensor],dim=0)