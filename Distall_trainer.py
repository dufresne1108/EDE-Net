import torch
from torchvision import transforms
from torch.utils import data
import utils

import numpy as np

import logging


from torch.nn import functional as F

class Trainer(object):
    def __init__(self, new_model,old_model,lr=0.0001,epoch=300,trainloader=None, testloader=None, unlabeled_trainloader=None,
                 num_class=10, save_path="",use_unlabeled=False):
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
        self.net = new_model
        self.use_unlabeled=use_unlabeled
        self.old_model = old_model
        self.old_model.eval()
        self.main_net_path = save_path
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=5e-4)
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

        print("Test set: Average  Accuracy top1:", correct_top1)
        return correct_top1, correct_top5


    def train(self):
        best_acc, _ = self.eval_training()
        print("init test acc:", best_acc)

        for epoch in range(self.epoch):

            if epoch > self.warm:
                self.train_scheduler.step()

            if self.use_unlabeled:
                train_loss = self.train_one_epoch_mixmatch()
            else:
                train_loss = self.train_one_epoch()

            curr_acc, _ = self.eval_training()

            info = "epoch:" + str(epoch) + " best_acc:" + str(best_acc) + " curr_acc:" + str(curr_acc) + " lr:" + str(
                self.train_scheduler.get_last_lr())
            logging.info(info)
            if curr_acc > best_acc:
                best_acc = curr_acc
                torch.save(
                    {'cfg': self.net.cfg, 'state_dict': self.net.state_dict(), 'best_acc': best_acc},
                    self.main_net_path)
                # torch.save({'cfg': self.net.cfg, 'state_dict': self.net.state_dict()}, self.main_net_path)

        print('Finished Training')
        return best_acc.item()

    def train_one_epoch(self):
        self.net.train()


        losses = utils.AvgrageMeter()

        for i, (inputs, labels) in enumerate(self.trainloader):


            batch_size = inputs.size(0)
            labels = torch.zeros(batch_size, self.num_class).scatter_(1, labels.view(-1, 1), 1)
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            all_inputs = torch.cat([inputs, inputs], dim=0)
            all_targets = torch.cat([labels, labels], dim=0)

            l = np.random.beta(self.alpha, self.alpha)
            l = max(l, 1 - l)
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation


            logits = self.net(mixed_input)

            Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
            with torch.no_grad():
                old_out = self.old_model(mixed_input)
            loss_dist = F.binary_cross_entropy(torch.softmax(logits / 2.0, dim=1),
                                               torch.softmax(old_out / 2.0, dim=1))


            loss = Lx + loss_dist

            losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            self.optimizer.step()

        return losses.avg

    def train_one_epoch_mixmatch(self):
        self.net.train()
        unlabeled_train_iter = iter(self.unlabeled_trainloader)

        losses = utils.AvgrageMeter()

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
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                # old_out_unlabel = self.old_model(inputs_u)
                # old_out = self.old_model(inputs)
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

            probs_u = torch.softmax(logits_u, dim=1)
            Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:size], dim=1))
            Lu = torch.mean((probs_u - mixed_target[size:]) ** 2)

            with torch.no_grad():
                old_out = self.old_model(torch.cat(mixed_input, dim=0))
            loss_dist = F.binary_cross_entropy(torch.softmax(torch.cat(logits, dim=0)/ 2.0 , dim=1),
                                               torch.softmax(old_out / 2.0, dim=1))
            # loss_dist_unlabel = F.binary_cross_entropy(torch.softmax(new_out_unlabel / 2.0, dim=1),
            #                              torch.softmax(old_out_unlabel / 2.0, dim=1))

            loss = Lx +loss_dist+ 10 * Lu

            losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            self.optimizer.step()

        return losses.avg



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










