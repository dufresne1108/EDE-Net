import torch

from torchvision import transforms
from torch.utils import data
import utils
import time
import os
from torch.nn import functional as F
import numpy as np

from cifar100_split import Cifar100Split
from cifar100_rehearsal import Cifar100Rehearsal
import logging
import sys
from models import EDE_Net
from ResNet import resnet18


class Trainer(object):
    def __init__(self, rehearsal_loader=None, current_data_loader=None, testloader=None, net=None, num_class=10,
                 save_path="", start_num=0, split_list=[], suffix=".ptn"):
        self.lr = 0.001
        self.epoch = 50
        self.use_cuda = True
        self.num_class = num_class
        self.save_path = save_path
        self.testloader = testloader
        self.rehearsal_loader = rehearsal_loader
        self.current_data_loader = current_data_loader
        self.T = 0.5
        self.alpha = 0.75
        self.net = net
        self.start_num = start_num
        self.split_list = split_list
        self.suffix = suffix
        params_list = []
        for feature in net.classifier_list:
            pp = {'params': feature.parameters()}
            params_list.append(pp)
        self.optimizer = torch.optim.Adam(params_list, lr=self.lr,weight_decay=5e-4)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)

    def eval_training(self):
        self.net.eval()
        for ff in self.net.classifier_list:
            ff.eval()

        correct_top1 = []

        for (inputs, labels) in self.testloader:
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.net(inputs)
                prec1 = self.accuracy(outputs.data,
                                      labels.cuda().data, topk=(1,))
            correct_top1.append(prec1[0].cpu().numpy())
        correct_top1 = sum(correct_top1) / len(correct_top1)
        print("Test set: Average  controller top1:", correct_top1)
        return correct_top1

    def train_one_epoch(self):
        batch_time = utils.AvgrageMeter()
        losses = utils.AvgrageMeter()
        end = time.time()
        train_iter = iter(self.current_data_loader)
        for i, (inputs, labels) in enumerate(self.rehearsal_loader):
            try:
                inputs_curr, labels_curr = train_iter.__next__()  # [64, 3, 32, 32]  [64, 3, 32, 32]
                labels_curr = labels_curr + self.start_num
            except:
                train_iter = iter(self.current_data_loader)
                inputs_curr, labels_curr = train_iter.__next__()
                labels_curr = labels_curr + self.start_num

            labels = torch.zeros(inputs.size(0), self.num_class).scatter_(1, labels.view(-1, 1), 1)
            labels_curr = torch.zeros(inputs_curr.size(0), self.num_class).scatter_(1, labels_curr.view(-1, 1), 1)
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.float().cuda()
                inputs_curr, labels_curr = inputs_curr.cuda(), labels_curr.float().cuda()

            all_inputs = torch.cat([inputs, inputs_curr], dim=0)
            all_targets = torch.cat([labels, labels_curr], dim=0)
            l = np.random.beta(self.alpha, self.alpha)
            l = max(l, 1 - l)
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = self.net(mixed_input)
            loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
            losses.update(loss.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
        return losses.avg

    def train(self):
        best_acc = self.eval_training()
        print("init test acc:", best_acc)
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            self.net.train()
            for ff in self.net.classifier_list:
                ff.train()
            _ = self.train_one_epoch()
            curr_acc = self.eval_training()
            self.train_scheduler.step()
            info = "epoch:" + str(epoch) + " best_acc:" + str(best_acc) + " curr_acc:" + str(curr_acc) + " lr:" + str(
                self.train_scheduler.get_last_lr()[0])
            logging.info(info)
            if curr_acc > best_acc:
                best_acc = curr_acc
                for i in range(len(self.net.classifier_list)):
                    model = self.net.classifier_list[i]
                    feature = {"state_dict": model.state_dict(), "cfg": model.cfg}
                    torch.save(feature, self.save_path + "/model_" + str(self.split_list[i]) + "to" + str(
                        self.split_list[i + 1]) + self.suffix)
        print('Finished Training')

    def accuracy(self, output, target, topk=(1,)):
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


def get_Rehearsal_data(start_num=0, end_num=10, batch_size=64, rehearsal_size=2000):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = Cifar100Split(start_num=start_num, end_num=end_num, train=True, transform=transform_train)

    rehearsal_train = Cifar100Rehearsal(end_num=start_num, rehearsal_size=rehearsal_size, transform=transform_train)

    testset = Cifar100Split(start_num=0, end_num=end_num, train=False, transform=transform_test)
    r_batch_size = int(batch_size * (start_num / end_num))
    c_batch_size = batch_size - r_batch_size
    current_data_loader = data.DataLoader(trainset, batch_size=c_batch_size, shuffle=True, num_workers=0,
                                          pin_memory=True)
    rehearsal_loader = data.DataLoader(rehearsal_train, batch_size=r_batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return rehearsal_loader, current_data_loader, testloader


def train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], split_type="s0-t10", prune=True,
                rehearsal_size=2000):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    model_type = "resnet18"
    save_path = "model/EDE_Net_cifar100_" + split_type + "_" + model_type + "_rehearsal_" + str(rehearsal_size)
    classifier_path = "model/classifier_cifar100_" + split_type + "_" + model_type + "_mixmatch"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    if prune:
        suffix = "_prune.ptn"
    else:
        suffix = ".ptn"
    n = len(split_list) - 2
    print(n)
    batch_size = 64
    test = True
    for i in range(0, n):
        torch.cuda.empty_cache()
        logging.info("train model %d" % (i + 1))
        classifier_list = []
        for j in range(i + 2):
            numclass = split_list[j + 1] - split_list[j]
            load_path = save_path + "/model_" + str(split_list[j]) + "to" + str(split_list[j + 1]) + suffix
            if os.path.exists(load_path):
                checkpoint = torch.load(load_path)
            else:
                load_path = classifier_path + "/model_" + str(split_list[j]) + "to" + str(split_list[j + 1]) + suffix
                checkpoint = torch.load(load_path)
            print(load_path)

            cfg = checkpoint['cfg']
            net1 = resnet18(num_class=numclass, cfg=cfg).cuda()
            net1.load_state_dict(checkpoint['state_dict'])
            classifier_list.append(net1)

        print("类别数量", split_list[i + 2])
        net = EDE_Net(classifier_list, class_num=split_list[i + 2]).cuda()
        r_loader, cur_loader, testloader = get_Rehearsal_data(start_num=split_list[i + 1],
                                                              end_num=split_list[i + 2],
                                                              batch_size=batch_size,
                                                              rehearsal_size=rehearsal_size)
        trainer = Trainer(net=net, rehearsal_loader=r_loader, current_data_loader=cur_loader,
                          testloader=testloader, num_class=split_list[i + 2],
                          save_path=save_path, start_num=split_list[i + 1], split_list=split_list, suffix=suffix)
        trainer.train()

        if test:
            classifier_list = []
            for j in range(i + 2):
                numclass = split_list[j + 1] - split_list[j]
                load_path = save_path + "/model_" + str(split_list[j]) + "to" + str(split_list[j + 1]) + suffix
                print(load_path)
                checkpoint = torch.load(load_path)
                cfg = checkpoint['cfg']
                net1 = resnet18(num_class=numclass, cfg=cfg).cuda()
                net1.load_state_dict(checkpoint['state_dict'])
                classifier_list.append(net1)
            net = EDE_Net(classifier_list, class_num=split_list[i + 2]).cuda()

            trainer = Trainer(net=net, rehearsal_loader=r_loader, current_data_loader=cur_loader,
                              testloader=testloader, num_class=split_list[i + 2],
                              save_path=save_path, start_num=split_list[i + 1], split_list=split_list, suffix=suffix)
            trainer.eval_training()


if __name__ == '__main__':
    train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                split_type="s0-t10", prune=True, rehearsal_size=2000)
    # train_model(split_list=[0, 20, 40, 60, 80, 100], split_type="s0-t5", rehearsal_size=2000)
    # train_model(split_list=[0, 50, 60, 70, 80, 90, 100], split_type="s50-t5", rehearsal_size=2000)
    # train_model(split_list=[0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], split_type="s50-t10", rehearsal_size=2000)
