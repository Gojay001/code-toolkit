import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import amp


# version 1: use torch.autograd
class FocalLossV1(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        """
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> label = torch.randint(0, 2, (8, 19, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        """

        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                                F.softplus(logits, -1, 50),
                                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                                  -logits + F.softplus(logits, -1, 50),
                                  -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.vars = (coeff, probs, log_probs, log_1_probs, probs_gamma,
                    probs_1_gamma, label, gamma)

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        (coeff, probs, log_probs, log_1_probs, probs_gamma,
         probs_1_gamma, label, gamma) = ctx.vars

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLossV2(nn.Module):
    """
    This use better formula to compute the gradient, which has better numeric stability
    Usage is like this:
        >>> criteria = FocalLossV2()
        >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)) # nchw, int64_t
        >>> loss = criteria(logits, lbs)
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        logits and label have same shape, and label data type is long
        args:
            predict: tensor of shape (N, ...)
            target: tensor of shape(N, ...)
        Usage is like this:
            >>> criterion = BinaryDiceLoss()
            >>> predict = torch.randn(8, 2, 384, 384)# nchw, float/half
            >>> target = torch.randint(0, 2, (8, 2, 384, 384)) # nchw, int64_t
            >>> loss = criterion(predict, target)
        """
        assert predict.shape[0] == target.shape[0]
        predict = torch.sigmoid(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - 2 * num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


if __name__ == '__main__':
    criterion1 = FocalLossV1()
    criterion2 = BinaryDiceLoss()
    preds = torch.randn(8, 2, 384, 384)  # NCHW, float/half
    label = torch.randint(0, 2, (8, 2, 384, 384))  # NCHW, int64_t
    loss1 = criterion1(preds, label)
    loss2 = criterion2(preds, label)
    print(loss1, loss2)
