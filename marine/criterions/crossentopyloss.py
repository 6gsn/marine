import torch
from torch.nn.modules.loss import _Loss


class CrossEntropyLoss(_Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=0)

    def forward(self, logits, labels, mask=None):
        loss = torch.zeros(1, device=logits.device)
        batch_size = logits.size(0)

        for logit, label, m in zip(logits, labels, mask):
            loss += self.loss_func(logit[m], label[m])

        return torch.sum(loss / batch_size)
