import torch
from torchvision import transforms
from torch.utils import data
import utils
import time
import os
import numpy as np
from ResNet import resnet18

from cifar100_split import Cifar100Split
from cifar10_svnh import Cifar10_SVNH_Split
import logging
import sys

from Distall_trainer import Trainer as dis_trainer
import PruneUtils




def get_data(start_num, end_num, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = Cifar100Split(start_num=start_num, end_num=end_num, train=True, transform=transform_train)
    testset = Cifar100Split(start_num=start_num, end_num=end_num, train=False, transform=transform_test)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_unlabeled_set = Cifar10_SVNH_Split(isCifar10=True, start_num=0, end_num=10, train=True,
                                             transform=transform_train)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=int(batch_size / 2), shuffle=True,
                                            num_workers=0, drop_last=True)

    return trainloader, testloader, unlabeled_trainloader




def eval_training(net,testloader):
    net.eval()
    correct_top1 = []

    for (inputs, labels) in testloader:


        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = net(inputs)

            prec1, prec5 = accuracy(outputs.data,
                                         labels.cuda().data, topk=(1, 5))
        correct_top1.append(prec1.cpu().numpy())

    correct_top1 = sum(correct_top1) / len(correct_top1)

    print("Test set: Average  Accuracy top1:", correct_top1)
    return correct_top1

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_classifier(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], split_type="s0-t10", mixmatch=True):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    model_type = "resnet18"
    if mixmatch is True:
        # save_path = "model/EDE_Net_cifar100_" + split_type + "_" + model_type
        save_path = "model/classifier_cifar100_" + split_type + "_" + model_type+"_mixmatch"

    else:
        save_path = "model/classifier_cifar100_" + split_type + "_" + model_type

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    n = len(split_list) - 1
    for i in range(0, n):
        torch.cuda.empty_cache()

        #
        logging.info("train model %d" % (i + 1))
        num_class = split_list[i + 1] - split_list[i]
        model_save_path = save_path + "/model_" + str(split_list[i]) + "to" + str(split_list[i + 1]) + ".ptn"
        checkpoint = torch.load(model_save_path)
        cfg = checkpoint['cfg']
        # acc = checkpoint['best_acc']
        # print("best_acc",acc)
        net = resnet18(cfg=cfg, num_class=num_class).cuda()
        net.load_state_dict(checkpoint['state_dict'])

        testset = Cifar100Split(start_num=split_list[i], end_num=split_list[i + 1], train=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
        best_acc_teacher = eval_training(net=net,testloader=testloader)
        print(best_acc_teacher)
        # 计算params,FLOPs
        print("old_model:")
        flops_num_old = PruneUtils.count_model_param_flops(model=net)
        para_num_old = PruneUtils.print_model_param_nums(model=net)




if __name__ == '__main__':
    # eval_classifier(split_list=[0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], split_type="s50-t10", mixmatch=True)
    eval_classifier(split_list=[0, 50, 60, 70, 80, 90, 100], split_type="s50-t5", mixmatch=True)