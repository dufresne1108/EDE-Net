import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import utils
import time
import os
from ResNet import resnet18
from ResNet import resnet34
from cifar10_svnh import Cifar10_SVNH_Split
import logging
import sys
from Rehearsal_MNIST_CIFAR_SVNH import Rehearsal_MNIST_CIFAR_SVNH
from models import EDE_Net


class Trainer(object):
    def __init__(self, epoch=50, trainloader=None, testloader=None, net=None, train_EDE=False, num_class=10,
                 save_path="", model_type=""):
        self.lr = 0.0001
        self.epoch = epoch
        self.warm = 1
        self.use_cuda = True
        self.num_class = num_class
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = net
        self.main_net_path = save_path
        self.train_EDE = train_EDE
        self.model_type = model_type
        if train_EDE:
            params_list = []
            for feature in net.classifier_list:
                pp = {'params': feature.parameters()}
                params_list.append(pp)

            self.optimizer = torch.optim.Adam(params_list, lr=self.lr, weight_decay=5e-4)
        else:
            self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=5e-4)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)

    def eval_training(self):
        self.net.eval()
        if self.train_EDE:
            for ff in self.net.classifier_list:
                ff.eval()
        correct_top1 = []
        correct_top5 = []
        for (inputs, labels) in self.testloader:

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.net(inputs)

                prec1 = self.accuracy(outputs.data,
                                      labels.cuda().data, topk=(1,))
            correct_top1.append(prec1[0].cpu().numpy())

        correct_top1 = sum(correct_top1) / len(correct_top1)

        print("Test set: Average  Accuracy top1:", correct_top1)
        return correct_top1, correct_top5

    def train_one_epoch(self):
        self.net.train()
        if self.train_EDE:
            for ff in self.net.classifier_list:
                ff.train()
        losses = utils.AvgrageMeter()

        for i, (inputs, labels) in enumerate(self.trainloader):

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            out = self.net(inputs)
            loss = torch.nn.functional.cross_entropy(out, labels)
            losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            self.optimizer.step()

        return losses.avg

    def train(self):
        best_acc, _ = self.eval_training()
        print("init test acc:", best_acc)

        for epoch in range(self.epoch):  # loop over the dataset multiple times

            train_loss = self.train_one_epoch()

            curr_acc, _ = self.eval_training()

            self.train_scheduler.step()
            info = self.model_type + "__epoch:" + str(epoch) + " best_acc:" + str(best_acc) + " curr_acc:" + str(
                curr_acc) + " lr:" + str(
                self.train_scheduler.get_last_lr()[0])
            logging.info(info)
            if curr_acc > best_acc:
                best_acc = curr_acc

                torch.save(self.net.state_dict(), self.main_net_path)

        print('Finished Training')

    def eval_EDE_Net(self, test_loader_list):
        self.net.eval()
        for ff in self.net.classifier_list:
            ff.eval()
        top1 = 0

        for i in range(len(test_loader_list)):
            loader = test_loader_list[i]
            correct_top1 = []

            for (inputs, labels) in loader:
                labels = labels + i * 10

                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                with torch.no_grad():
                    outputs = self.net(inputs)

                    prec1 = self.accuracy(outputs.data,
                                          labels.cuda().data, topk=(1,))

                correct_top1.append(prec1[0].cpu().numpy())
                # correct_top12.append(prec3[0].cpu().numpy())
            tem_top1 = sum(correct_top1) / len(correct_top1)
            top1 += tem_top1

        correct_top1 = top1 / len(test_loader_list)
        print("Test set: Average  controller top1:", correct_top1)

        return correct_top1

    def train_EDE_Net_one_epoch(self, rehearsal_loader):

        batch_time = utils.AvgrageMeter()

        losses = utils.AvgrageMeter()
        for ff in self.net.classifier_list:
            ff.train()
        end = time.time()

        for i, (inputs, labels) in enumerate(rehearsal_loader):

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # print(labels)
            class_out = self.net(inputs)
            # print(labels)
            loss = torch.nn.functional.cross_entropy(class_out, labels)

            losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)

        return losses.avg

    def train_EDE_Net(self, train_loader, test_loader_list):
        best_acc = self.eval_EDE_Net(test_loader_list)
        print("init test acc:", best_acc)

        for epoch in range(self.epoch):  # loop over the dataset multiple times

            train_loss = self.train_EDE_Net_one_epoch(train_loader)
            curr_acc = self.eval_EDE_Net(test_loader_list)
            self.train_scheduler.step()
            info = self.model_type + "__epoch:" + str(epoch) + " best_acc:" + str(best_acc) + " curr_acc:" + str(
                curr_acc) + " lr:" + str(
                self.train_scheduler.get_last_lr()[0])
            logging.info(info)
            if curr_acc > best_acc:
                best_acc = curr_acc

                torch.save(self.net.state_dict(), self.main_net_path)

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_data(data_type="cifar10", batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if data_type is "cifar10":

        trainset = Cifar10_SVNH_Split(isCifar10=True, start_num=0, end_num=10, train=True,
                                      transform=transform_train)
        testset = Cifar10_SVNH_Split(isCifar10=True, start_num=0, end_num=10, train=False,
                                     transform=transform_test)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


    elif data_type is "SVNH":

        trainset = Cifar10_SVNH_Split(isCifar10=False, start_num=0, end_num=10, train=True,
                                      transform=transform_train)
        testset = Cifar10_SVNH_Split(isCifar10=False, start_num=0, end_num=10, train=False,
                                     transform=transform_test)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    elif data_type is "MNIST":
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            to3channels(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            to3channels(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            transform=transform_test,
            download=True,
        )

        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


def get_rehearsal_data(batch_size=64, data_type_list=["MNIST", "SVNH", 'cifar10'],
                       file_path="data/", num_per_class=50, ):
    trans_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trans_MNIST = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        to3channels(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    rehearasl = Rehearsal_MNIST_CIFAR_SVNH(data_type_list=data_type_list,
                                           file_path=file_path, num_per_class=num_per_class,
                                           train=True, transform_list=[trans_MNIST, trans_cifar10, trans_cifar10])
    rehearasl_loader = data.DataLoader(rehearasl, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_list = []
    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        if data_type is "cifar10":
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            Cifar10_testset = Cifar10_SVNH_Split(isCifar10=True, start_num=0, end_num=10, train=False,
                                                 transform=transform_test)
            testloader = data.DataLoader(Cifar10_testset, batch_size=batch_size, shuffle=False, num_workers=0)

        elif data_type is "SVNH":
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            SVNH_testset = Cifar10_SVNH_Split(isCifar10=False, start_num=0, end_num=10, train=False,
                                              transform=transform_test)

            testloader = data.DataLoader(SVNH_testset, batch_size=batch_size, shuffle=False, num_workers=0)

        elif data_type is "MNIST":
            transform_test_MNIST = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                to3channels(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            MNIST_testset = torchvision.datasets.MNIST(
                root=file_path,
                train=False,
                transform=transform_test_MNIST,
                download=True,
            )
            testloader = data.DataLoader(MNIST_testset, batch_size=batch_size, shuffle=False, num_workers=0)

        test_loader_list.append(testloader)

    return rehearasl_loader, test_loader_list


class to3channels(object):

    def __call__(self, tensor):
        return torch.cat([tensor, tensor, tensor], dim=0)


def train_MTIL_classifier(data_type="MNIST", num_class=10, model_type="resnet18", batch_size=128):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    save_path = "model/MTIL/classifier_" + model_type
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    torch.cuda.empty_cache()

    trainloader, testloader = get_data(data_type=data_type, batch_size=batch_size)

    net = resnet18(cfg=[64, 128, 128, 128], num_class=10).cuda()
    model_save_path = save_path + "/model_" + data_type + ".ptn"
    trainer = Trainer(epoch=150, net=net, trainloader=trainloader, testloader=testloader, num_class=num_class,
                      save_path=model_save_path, model_type=data_type)
    trainer.train()
    # prev_best = torch.load(model_save_path)
    # net.load_state_dict(prev_best)
    # trainer.eval_training()


def train_EDE_Net(data_list=["MNIST", "SVNH", 'cifar10'], train_EDE=True, file_path="./data", num_class=10,
                  model_type="resnet18",
                  num_per_class=20, batch_size=128):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    save_path = "model/MTIL/EDE_Net_" + model_type
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    torch.cuda.empty_cache()

    rehearsal_loader, test_loader_list = get_rehearsal_data(data_type_list=data_list, file_path=file_path,
                                                            batch_size=batch_size, num_per_class=num_per_class)

    classifier_list = []
    for i in range(len(data_list)):
        net1 = resnet18(cfg=[64, 128, 128, 128], num_class=10).cuda()
        load_path = "model/MTIL/classifier_" + model_type + "/model_" + data_list[i] + ".ptn"
        print(load_path)
        prev_best = torch.load(load_path)
        net1.load_state_dict(prev_best)
        classifier_list.append(net1)

    net = EDE_Net(classifier_list, class_num=len(classifier_list) * 10).cuda()

    name = ""
    for item in data_list:
        name += item + "_"
    model_save_path = save_path + "/model_" + name + str(num_per_class) + ".ptn"
    print(model_save_path)
    # prev_best = torch.load(model_save_path)
    # net.load_state_dict(prev_best)

    trainer = Trainer(epoch=20, net=net, num_class=num_class, train_EDE=train_EDE, model_type=name,
                      save_path=model_save_path)
    trainer.train_EDE_Net(train_loader=rehearsal_loader, test_loader_list=test_loader_list)
    # prev_best = torch.load(model_save_path)
    # net.load_state_dict(prev_best)
    # trainer.eval_controller(test_loader_list=test_loader_list)


if __name__ == '__main__':
    train_MTIL_classifier(data_type="MNIST", batch_size=128)
    train_MTIL_classifier(data_type="SVNH", batch_size=128)
    train_MTIL_classifier(data_type="cifar10", batch_size=128)
    train_EDE_Net(data_list=["MNIST", "SVNH"], num_per_class=1)
    train_EDE_Net(data_list=["MNIST", "SVNH", 'cifar10'], num_per_class=1)
    train_EDE_Net(data_list=["MNIST", "SVNH"], num_per_class=5)
    train_EDE_Net(data_list=["MNIST", "SVNH", 'cifar10'], num_per_class=5)
    train_EDE_Net(data_list=["MNIST", "SVNH", ], num_per_class=20)
    train_EDE_Net(data_list=["MNIST", "SVNH", 'cifar10'], num_per_class=20)
    train_EDE_Net(data_list=["MNIST", "SVNH", ], num_per_class=50)
    train_EDE_Net(data_list=["MNIST", "SVNH", 'cifar10'], num_per_class=50)
