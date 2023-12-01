import torch
import torch.nn as nn
from .self_attention import SelfAttention
from .cross_attention import CrossAttention
from argument import Option


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.sa = SelfAttention(d_model=args.d_model,  pretrained=args.pretrained)
        self.ca = CrossAttention(args=args, h=args.head, n=args.number,
                                 d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)

    def forward(self, sk, im, stage='train'):
        if stage == 'train':
            sk_im = torch.cat((sk, im), dim=0)
            sa_feature, final_tokens, inds = self.sa(sk_im)  # [4b, 197, 768]
            ca_feature = self.ca(sa_feature)  # [4b, 197, 768]
            cls_feature = ca_feature[:, 0]  # [4b, 1, 768]
            # print('cls_feature:', cls_feature.size())

            return cls_feature

        else:
            sa_feature, final_tokens, inds = self.sa(sk)
            return sa_feature, inds


if __name__ == '__main__':
    args = Option().parse()
    sk = torch.rand((10, 3, 224, 224))
    im = torch.rand((10, 3, 224, 224))
    model = Model(args)
    cls_fea = model(sk, im)
