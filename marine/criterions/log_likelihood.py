from torch.nn.modules.loss import _Loss


class LogLikelhood(_Loss):
    def __init__(self, log_likehood_func):
        super(LogLikelhood, self).__init__()
        self.log_likehood_func = log_likehood_func

    def forward(self, classified, label, mask=None):
        batch_size = label.size(0)
        log_likelihood = self.log_likehood_func(classified, label, mask)

        return -log_likelihood / batch_size
