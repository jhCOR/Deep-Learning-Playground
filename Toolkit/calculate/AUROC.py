from sklearn.metrics import roc_auc_score
import torch

class AUROC():
    def __init__(self):
        self.auroc_fun = roc_auc_score()
        self.pred = None
        self.label = None
        self.device = "cpu"
        
    def to(self, device):
        self.device = device

    def update(self, pred, label):
        self.pred = pred if self.pred is None else torch.cat([self.pred, pred])
        self.label = label if self.label is None else torch.cat([self.label, label])

    def compute(self):
        return self.auroc_fun(self.pred.to(self.device), self.label.to(self.device))
    
    def reset(self):
        self.pred = None
        self.label = None