import torch
from torchvision import transforms
from torch.utils import data
import utils
import os
import numpy as np
from ResNet import resnet18
from cifar100_split import Cifar100Split
from cifar10_svnh import Cifar10_SVNH_Split
import logging
import sys
from Distall_trainer import Trainer as dis_trainer
import PruneUtils


class Trainer(object):
    def __init__(self, trainloader=None, lr=0.001, epoch=300, testloader=None, unlabeled_trainloader=None, net=None,
                 num_class=10, save_path="", mixmatch=True):
        self.lr = lr
        self.epoch = epoch
        self.warm = 1
        self.use_cuda = True
        self.num_class = num_class
        self.unlabeled_trainloader = unlabeled_trainloader
        self.trainloader = trainloader
        self.testloader = testloader
        self.T = 0.5
        self.alpha = 0.75
        self.net = net
        self.use_mixmatch = mixmatch
        self.main_net_path = save_path
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=5e-4)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)

    def eval_training(self):
        self.net.eval()
        correct_top1 = []
        correct_top5 = []
        for (inputs, labels) in self.testloader:

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.net(inputs)

                prec1, prec5 = self.accuracy(outputs.data,
                                             labels.cuda().data, topk=(1, 5))
            correct_top1.append(prec1.cpu().numpy())

        correct_top1 = sum(correct_top1) / len(correct_top1)

        return correct_top1, correct_top5

    def train_one_epoch_mixmatch(self):
        self.net.train()
        unlabeled_train_iter = iter(self.unlabeled_trainloader)

        criterion = self.SemiLoss()

        losses = utils.AvgrageMeter()
        losses_x = utils.AvgrageMeter()
        losses_u = utils.AvgrageMeter()

        for i, (inputs, labels) in enumerate(self.trainloader):

            try:
                inputs_u, _ = unlabeled_train_iter.__next__()  # [64, 3, 32, 32]  [64, 3, 32, 32]
                inputs_u2, _ = unlabeled_train_iter.__next__()
            except:
                unlabeled_train_iter = iter(self.unlabeled_trainloader)
                inputs_u, _ = unlabeled_train_iter.__next__()
                inputs_u2, _ = unlabeled_train_iter.__next__()
            batch_size = inputs.size(0)
            labels = torch.zeros(batch_size, self.num_class).scatter_(1, labels.view(-1, 1), 1)
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.float().cuda()
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u = self.net(inputs_u)
                outputs_u2 = self.net(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p ** (1 / self.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            all_inputs = torch.cat([inputs, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([labels, targets_u, targets_u], dim=0)

            l = np.random.beta(self.alpha, self.alpha)
            l = max(l, 1 - l)
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = self.interleave(mixed_input, batch_size)

            logits = [self.net(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(self.net(input))

            logits = self.interleave(logits, batch_size)

            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)  # [128, 10]

            size = logits_x.size(0)
            Lx, Lu = criterion(logits_x, mixed_target[:size], logits_u, mixed_target[size:])

            loss = Lx + 10 * Lu

            losses.update(loss.item(), inputs.size(0))
            losses_x.update(Lx.item(), inputs.size(0))
            losses_u.update(Lu.item(), inputs.size(0))

            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            self.optimizer.step()

        return (losses.avg, losses_x.avg, losses_u.avg,)

    def train_one_epoch(self):
        self.net.train()

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

        for epoch in range(self.epoch):

            if epoch > self.warm:
                self.train_scheduler.step()

            if self.use_mixmatch:
                train_loss, train_loss_x, train_loss_u = self.train_one_epoch_mixmatch()
            else:
                train_loss = self.train_one_epoch()

            curr_acc, _ = self.eval_training()
            # if epoch %5==0:
            #     print("Test set: Average  Accuracy top1:", curr_acc)

            info = "epoch:" + str(epoch) + " best_acc:" + str(best_acc) + " curr_acc:" + str(curr_acc) + " lr:" + str(
                self.train_scheduler.get_last_lr())
            logging.info(info)
            if curr_acc > best_acc:
                best_acc = curr_acc
                torch.save({'cfg': self.net.cfg, 'state_dict': self.net.state_dict(), 'best_acc': best_acc},
                           self.main_net_path)
        print('Finished Training')
        return best_acc.item()

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

    class SemiLoss(object):
        def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
            probs_u = torch.softmax(outputs_u, dim=1)

            Lx = -torch.mean(torch.sum(torch.nn.functional.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
            Lu = torch.mean((probs_u - targets_u) ** 2)

            return Lx, Lu

    class WeightEMA(object):
        def __init__(self, model, ema_model, lr=0.002, alpha=0.999):
            self.model = model
            self.ema_model = ema_model
            self.alpha = alpha
            self.params = list(model.state_dict().values())
            self.ema_params = list(ema_model.state_dict().values())
            self.wd = 0.02 * lr

            for param, ema_param in zip(self.params, self.ema_params):
                param.data.copy_(ema_param.data)

        def step(self):
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.params, self.ema_params):
                if ema_param.dtype == torch.float32:
                    ema_param.mul_(self.alpha)
                    ema_param.add_(param * one_minus_alpha)

                    param.mul_(1 - self.wd)

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


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
    # trainset = Cifar100Rehearsal(end_num=end_num, rehearsal_size=2000, transform=transform_train)

    testset = Cifar100Split(start_num=start_num, end_num=end_num, train=False, transform=transform_test)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    train_unlabeled_set = Cifar10_SVNH_Split(isCifar10=True, start_num=0, end_num=10, train=True,
                                             transform=transform_train)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=int(batch_size / 2), shuffle=True,
                                            num_workers=0, drop_last=True, pin_memory=True)

    return trainloader, testloader, unlabeled_trainloader


def get_model(num_class=10, model_type="resnet18"):
    if model_type is "resnet18":
        model = resnet18(cfg=[64, 128, 256, 512], num_class=num_class).cuda()

    elif model_type is "resnet34":
        model = None
    else:
        model = None
    return model


def train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], split_type="s0-t10", prune=True,
                mixmatch=True):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    model_type = "resnet18"
    if mixmatch is True:
        save_path = "model/classifier_cifar100_" + split_type + "_" + model_type + "_mixmatch"

    else:
        save_path = "model/classifier_cifar100_" + split_type + "_" + model_type

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    n = len(split_list) - 1
    max_acc_loss = 4
    lr = 0.0005
    epoch = 400

    for i in range(0, n):
        torch.cuda.empty_cache()
        logging.info("train model %d" % (i + 1))
        num_class = split_list[i + 1] - split_list[i]
        net = get_model(num_class=num_class, model_type=model_type)
        _ = PruneUtils.print_model_param_nums(model=net)
        #
        trainloader, testloader, unlabeled_trainloader = get_data(start_num=split_list[i], end_num=split_list[i + 1],
                                                                  batch_size=128)
        #
        # 模型训练
        model_save_path = save_path + "/model_" + str(split_list[i]) + "to" + str(split_list[i + 1]) + ".ptn"
        trainer = Trainer(net=net, lr=lr, epoch=epoch, trainloader=trainloader, testloader=testloader,
                          unlabeled_trainloader=unlabeled_trainloader, num_class=num_class, mixmatch=mixmatch,
                          save_path=model_save_path)
        best_acc_teacher = trainer.train()

        if prune:
            # 计算params,FLOPs
            print("old_model:")
            checkpoint = torch.load(model_save_path)
            net.load_state_dict(checkpoint['state_dict'])
            best_acc_teacher = checkpoint['best_acc']
            _ = PruneUtils.count_model_param_flops(model=net)
            para_num_old = PruneUtils.print_model_param_nums(model=net)
            # 模型剪枝
            teacher_net = net
            temp_net = net
            pre_cfg = None
            while True:
                # 蒸馏训练剪枝模型
                cur_cfg = PruneUtils.prune_ResNet_channel(temp_net, percent=0.5)
                prune_model = resnet18(num_class=num_class, cfg=cur_cfg).cuda()

                print("prune_model:")
                _ = PruneUtils.count_model_param_flops(model=prune_model)
                para_num_prune = PruneUtils.print_model_param_nums(model=prune_model)
                print("training prune_model:")
                prune_save_path = save_path + "/model_" + str(split_list[i]) + "to" + str(
                    split_list[i + 1]) + "_prune.ptn"
                prune_temp_path = save_path + "/model_" + str(split_list[i]) + "to" + str(
                    split_list[i + 1]) + "_temp_prune.ptn"
                trainer = dis_trainer(new_model=prune_model, old_model=teacher_net, lr=lr, epoch=epoch,
                                      trainloader=trainloader, testloader=testloader,
                                      unlabeled_trainloader=unlabeled_trainloader, num_class=num_class,
                                      use_unlabeled=mixmatch,
                                      save_path=prune_temp_path)

                best_acc_student = trainer.train()
                if (best_acc_teacher - best_acc_student) > max_acc_loss:
                    break
                else:
                    pre_cfg = cur_cfg
                    checkpoint = torch.load(prune_temp_path)
                    torch.save(checkpoint, prune_save_path)
                    temp_net = prune_model

            # 再压缩
            if pre_cfg:
                cfg = [(pre_cfg[i] + cur_cfg[i]) // 2 for i in range(len(pre_cfg))]
                prune_model = resnet18(num_class=num_class, cfg=cfg).cuda()
                para_num_prune = PruneUtils.print_model_param_nums(model=prune_model)
                print("training  prune_model:")
                trainer = dis_trainer(new_model=prune_model, old_model=teacher_net, lr=lr, epoch=epoch,
                                      trainloader=trainloader, testloader=testloader,
                                      unlabeled_trainloader=unlabeled_trainloader, num_class=num_class,
                                      use_unlabeled=mixmatch,
                                      save_path=prune_temp_path)
                best_acc_student = trainer.train()
                if (best_acc_teacher - best_acc_student) < max_acc_loss:
                    checkpoint = torch.load(prune_temp_path)
                    torch.save(checkpoint, prune_save_path)


if __name__ == '__main__':
    train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                split_type="s0-t10", prune=True, mixmatch=True)

    # train_model(split_list=[0, 20, 40, 60, 80, 100], split_type="s0-t5", mixmatch=True)

    # train_model(split_list=[0, 50, 60, 70, 80, 90, 100], split_type="s50-t5", mixmatch=True)
    # train_model(split_list=[0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], split_type="s50-t10", mixmatch=True)
