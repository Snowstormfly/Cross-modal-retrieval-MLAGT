import torch
import torch.nn as nn
import torch.nn.functional as F


def triplet_loss(x, args):
    """
     x: 4*batch -> sk_p, sk_n, im_p, im_n
    """
    triplet = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    sk_p = x[0:args.batch]
    im_p = x[2 * args.batch:3 * args.batch]
    im_n = x[3 * args.batch:]
    loss = triplet(sk_p, im_p, im_n)

    return loss


class CosineTripletLoss(nn.Module):
    """余弦相似度三元组损失函数"""
    def __init__(self, margin=1.0):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 归一化样本向量
        anchor = F.normalize(anchor)
        positive = F.normalize(positive)
        negative = F.normalize(negative)

        # 计算余弦相似度
        similarity = F.cosine_similarity(anchor, positive) - F.cosine_similarity(anchor, negative)

        # 计算损失
        loss = torch.clamp(similarity + self.margin, min=0.0).mean()

        return loss


def classify_loss(predict, target):
    class_loss = nn.CrossEntropyLoss().cuda()
    loss = class_loss(predict, target)

    return loss



