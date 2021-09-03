import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable


class AngularIsoLoss(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(AngularIsoLoss, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        # loss = self.softplus(torch.logsumexp(self.alpha * scores, dim=0))

        # loss = self.softplus(torch.logsumexp(self.alpha * scores[labels == 0], dim=0)) + \
        #        self.softplus(torch.logsumexp(self.alpha * scores[labels == 1], dim=0))

        loss = self.softplus(self.alpha * scores).mean()

        return loss, -output_scores.squeeze(1)

class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, -output_scores.squeeze(1)

class AMSoftmax(nn.Module):
    def __init__(self, num_classes, enc_dim, s=20, m=0.9):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits

if __name__ == "__main__":
    # feats = torch.randn((32, 90)).cuda()
    # centers = torch.randn((3,90)).cuda()
    # # o = torch.norm(feats - center, p=2, dim=1)
    # # print(o.shape)
    # # dist = torch.cat((o, o), dim=1)
    # # print(dist.shape)
    # labels = torch.cat((torch.Tensor([0]).repeat(10),
    #                    torch.Tensor([1]).repeat(22)),0).cuda()
    # # classes = torch.arange(2).long().cuda()
    # # labels = labels.expand(32, 2)
    # # print(labels)
    # # mask = labels.eq(classes.expand(32, 2))
    # # print(mask)
    #
    # iso_loss = MultiCenterIsolateLoss(centers, 2, 90).cuda()
    # loss = iso_loss(feats, labels)
    # for p in iso_loss.parameters():
    #     print(p)
    # # print(loss.shape)

    feat_dim = 16
    feats = torch.randn((32, feat_dim))
    labels = torch.cat((torch.Tensor([0]).repeat(10),
                        torch.Tensor([1]).repeat(22)), 0).cuda()
    aisoloss = AngularIsoLoss(feat_dim=feat_dim)
    loss = aisoloss(feats, labels)
    print(loss)