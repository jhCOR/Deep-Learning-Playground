import torch

class Loss():
    def __init__(self):
        self.loss_fun = torch.nn.BCELoss()
        self.pred = None
        self.label = None
        self.device = "cpu"

    def to(self, device):
        self.device = device

    def update(self, pred, label):
        self.pred = pred
        self.label = label

    def compute(self):
        return self.loss_fun(self.pred.to(self.device), self.label.to(self.device))
    
    def reset(self):
        self.pred = None
        self.label = None