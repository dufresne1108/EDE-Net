import torch.nn as nn

import torch


class EDE_Net(nn.Module):
    def __init__(self, classifier_list=[], class_num=10):
        super(EDE_Net, self).__init__()
        self.classifier_list = classifier_list
        input_dim = 0
        for net in classifier_list:
            input_dim += net.cfg[-1]
        self.class_num = class_num

    def forward(self, x):
        out_list = []
        for net in self.classifier_list:
            out = net(x)
            out_list.append(out)

        out = torch.cat(out_list, dim=1)
        return out

    def fixed_feature_forward(self, x):
        out_list = []
        for net in self.classifier_list:
            out = net.fixed_feature_forward(x, fixed=True)
            out_list.append(out)

        out = torch.cat(out_list, dim=1)
        return out


