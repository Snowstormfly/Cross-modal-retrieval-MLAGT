import torch
from torch import nn
from einops import rearrange, repeat
import math


MIN_NUM_PATCHES = 16


class Multi_Level_Extract(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 768, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        return self.seq(x)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim / heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, keep_rate, mask=None):
        b, n, c, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attention = dots.softmax(dim=-1)

        output = torch.einsum('bhij,bhjd->bhid', attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.to_out(output)

        final_tokens = n-1
        if keep_rate < 1:
            final_tokens = math.ceil(keep_rate * final_tokens)
            class_attention = attention[:, :, 0, 1:]
            class_attention = class_attention.mean(dim=1)
            _, ind = torch.topk(class_attention, final_tokens, dim=1, largest=True, sorted=True)  # [B, final_tokens]
            ind, _ = torch.sort(ind)
            index = ind.unsqueeze(-1).expand(-1, -1, c)  # [B, final_tokens, C]

            return x, index, ind, class_attention, final_tokens

        return output, None, None, None, final_tokens


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1, attention_dropout_rate=0.1,
                 deterministic=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.deterministic = deterministic
        self.input_shape = input_shape
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        self.attention = MultiHeadDotProductAttention(input_shape, heads=heads)
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention = nn.Dropout(attention_dropout_rate)

    def forward(self, inputs, keep_rate):
        x = self.layer_norm_input(inputs)
        x, index, ind, class_attention, final_tokens = self.attention(x, keep_rate)
        x = self.drop_out_attention(x)
        x = x + inputs
        
        # B, N, C = x.shape
        if index is not None:
            non_class = x[:, 1:]
            x_others = torch.gather(non_class, dim=1, index=index)    # [B, final_tokens, C]
            x = torch.cat([x[:, 0:1], x_others], dim=1)     # [B, N+1, C] ->  [B, final_tokens+1, C]

        y = self.layer_norm_out(x)
        y = self.mlp(y)
        
        return x + y, final_tokens, ind
    

class Encoder(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, inputs_positions=None, dropout_rate=0.1, train=True):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate
        self.train_flag = train
        self.encoder_norm = nn.LayerNorm(input_shape)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Encoder1DBlock(input_shape, heads, mlp_dim)]))

        self.keep_rate = (1, ) * 12

    def forward(self, img):
        x = img
        final_tokens = []
        inds = []

        for i, layer in enumerate(self.layers):
            x, final_token, ind = layer[0](x, self.keep_rate[i])
            final_tokens.append(final_token)
            inds.append(ind)

        return self.encoder_norm(x), final_tokens, inds


class ViTPatch(nn.Module):
    def __init__(self, *, image_size, patch_size, hidden_size, depth, heads, mlp_dim,
                 channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Conv2d(channels, hidden_size, patch_size, patch_size)
        self.scale = Multi_Level_Extract()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Encoder(hidden_size, depth, heads, mlp_dim, dropout_rate=dropout)
        self.to_cls_token = nn.Identity()

    def forward(self, img):
        x1 = self.embedding(img)
        x2 = self.scale(img)
        x = (x1 + x2) / 2

        x = rearrange(x, 'b c h w  -> b (h w) c')
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x, final_tokens, inds = self.transformer(x)

        return x, final_tokens, inds


class SelfAttention(nn.Module):
    def __init__(self, d_model=768, pretrained=True):
        super(SelfAttention, self).__init__()
        self.model = ViTPatch(
            image_size=224,
            patch_size=16,
            hidden_size=d_model,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1
        )
        if pretrained:
            checkpoint = torch.load(r'./model/sam_ViT-B_16.pth')
            cur = self.model.state_dict()
            new = {k: v for k, v in checkpoint.items() if k in cur.keys() and 'mlp_head' not in k}
            cur.update(new)
            self.model.load_state_dict(cur)

    def forward(self, x):
        sa_feature, final_tokens, inds = self.model(x)
        return sa_feature, final_tokens, inds

