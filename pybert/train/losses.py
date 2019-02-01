#encoding:utf-8
from torch.nn import CrossEntropyLoss

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()
    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss
