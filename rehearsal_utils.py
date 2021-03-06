
import numpy as np
import pickle

def create_split_train(file_path="data/cifar-100-python/", start_num=0, end_num=10, one_hot=False):
    data_dict = load_cifar100(file_path=file_path, one_hot=False)
    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    y_train = np.array(y_train)
    num_class = end_num - start_num
    a1 = y_train >= start_num
    a2 = y_train < end_num
    index = a1 & a2
    task_train_x = x_train[index]
    label = y_train[index]
    label = label - start_num
    if one_hot is True:
        task_train_y = np.zeros([task_train_x.shape[0], num_class])
        task_train_y[range(task_train_x.shape[0]), label] = 1
    else:
        task_train_y = label

    task_split = {
        "x": task_train_x,
        "y": task_train_y,
    }

    return task_split


def create_split_test( file_path="data/cifar-100-python/", start_num=0, end_num=10, one_hot=False):
    data_dict = load_cifar100(file_path=file_path, one_hot=False)

    x_test = data_dict['x_test']
    y_test = data_dict['y_test']
    y_test = np.array(y_test)
    num_class = end_num - start_num

    a1 = y_test >= start_num
    a2 = y_test < end_num
    index = a1 & a2
    task_test_x = x_test[index]
    label = y_test[index]
    label = label - start_num
    if one_hot is True:
        task_test_y = np.zeros([task_test_x.shape[0], num_class])
        task_test_y[range(task_test_x.shape[0]), label] = 1
    else:
        task_test_y = label
    task_split = {
        "x": task_test_x,
        "y": task_test_y
    }

    return task_split


def load_cifar100(file_path="data/cifar-100-python/", one_hot=True):
    # 读取训练数据
    with open(file_path + "train", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        x_train = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
        y_temp = dict[b'fine_labels']
    if one_hot is True:
        y_train = np.zeros([x_train.shape[0], 100])
        y_train[range(x_train.shape[0]), y_temp] = 1
    else:
        y_train = y_temp
    # 读取测试数据
    with open(file_path + "test", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        x_test = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
        y_temp = dict[b'fine_labels']
    if one_hot is True:
        y_test = np.zeros([x_test.shape[0], 100])
        y_test[range(x_test.shape[0]), y_temp] = 1
    else:
        y_test = y_temp

    # x_train, x_test = normalize(x_train, x_test)
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test
    }
    return data



def produce_rehearsal(file_path="data/cifar-100-python/", end_num=10,num_per_class=100):
    data_dict = load_cifar100(file_path=file_path, one_hot=False)
    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    y_train = np.array(y_train)
    rehearsal_data = None
    rehearsal_label = None
    for i in range(1,end_num):
        a1 = y_train >= i-1
        a2 = y_train < i
        index = a1 & a2
        task_train_x = x_train[index]
        label = y_train[index]
        index= np.random.choice(task_train_x.shape[0],num_per_class)
        tem_data = task_train_x[index]
        tem_label = label[index]
        if rehearsal_data is None:
            rehearsal_data =tem_data
            rehearsal_label =tem_label
        else:
            rehearsal_data = np.concatenate([rehearsal_data,tem_data],axis=0)
            rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

    task_split = {
        "x": rehearsal_data,
        "y": rehearsal_label,
    }
    return task_split

if __name__ == '__main__':
    produce_rehearsal()