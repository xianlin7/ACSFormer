from cgi import print_form
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torchvision.ops import roi_align, nms
from utils.bbox_tools import prior2bbox
from utils.visualization import visual_class_roi, visual_refer_region
import numpy as np

def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention_query_response(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, response):
        q = self.to_q(query)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(response).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        #attn = softmax_one(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_query_response(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2(dim, Attention_query_response(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, query, response):
        for attn, ff in self.layers:
            query = attn(query, response) + query
            query = ff(query) + query
        return query


class Intra_slice_Classaware_Interaction(nn.Module):
    def __init__(self, n_channels, n_classes=2, n_space = 8):
        super(Intra_slice_Classaware_Interaction, self).__init__()
        self.sig = nn.Sigmoid()
        
        self.to_cspace = nn.Conv2d(1, n_space-1, kernel_size=3, padding=1, bias=False)
        self.to_xspace = nn.Conv2d(n_channels, n_channels*n_space, kernel_size=3, padding=1, bias=False)
        self.combine_space = nn.Conv2d(n_space, 1, kernel_size=1, bias=False)

        self.to_cvalue =  nn.Conv2d(1, n_space, kernel_size=1, bias=False)
        self.to_attn = nn.Conv2d(1, n_space, kernel_size=1, bias=False)
        self.combine = nn.Conv2d(n_space, 1, kernel_size=1, bias=False)
        self.n_space = n_space

        self.single_conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, class_score):
        # generate class token through class-aware feature fusion 
        class_prob = self.sig(class_score) # b c h w
        space0 = rearrange(class_prob, 'b (c d) h w -> (b c) d h w', d=1) # [bc 1 h w]
        space_other = self.sig(self.to_cspace(space0)) # [bc s-1 h w]
        space = torch.cat([space0, space_other], dim=1) # [bc s h w]
        space_sum = torch.sum(torch.sum(space, dim=-1), dim=-1) # [bc s]
        space_norm = space/space_sum[:, :, None, None] # [bc s h w]
        space_norm = rearrange(space_norm, '(b c) s h w -> b s c h w', b=x.shape[0])
        xv = rearrange(self.to_xspace(x), 'b (s d) h w -> b s d h w', s=self.n_space)
        class_token = xv[:, :, None, :, :, :] * space_norm[:, :, :, None, :, :] # [b s c d h w]
        class_token = torch.sum(torch.sum(class_token, dim=-1), dim=-1) # [b s c d]
        class_token = self.combine_space(class_token).squeeze(dim=1) # [b c d] 

        class_value = self.to_cvalue(class_token[:, None, :, :]) # [b s c d]
        class_attn = rearrange(class_prob, 'b (c d) h w -> b d (h w) c', d=1)
        class_attn = self.sig(self.to_attn(class_attn)) # [b s N c]
        class_info = torch.matmul(class_attn, class_value) # [b s N d]
        class_info = self.combine(class_info).squeeze(dim=1) # [b N d]
        class_info = rearrange(class_info, 'b (h w) d -> b d h w', h=x.shape[2])

        class_info = self.single_conv(class_info)

        x = x + class_info
        return x


class Inter_slice_Adaptive_Interaction(nn.Module):
    """
    build long-range dependency cross the slices
    input:
        -- mslice: [b c h w] , the major slice for segmenting objects
        -- aslice: [k b c h w], the assist slices for segmenting objects
    """
    def __init__(self, in_channels, out_channels, num_classes, depth=2, patch_size=2, heads=6, assist_slice_number=4, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.patch_height, self.patch_width = pair(patch_size)
        #self.to_class_c = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.init_size = 5 #default = 9
        self.feature_size = 5 # default=5
        self.num_classes = num_classes
        self.assist_slice_number = assist_slice_number
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4
        self.num_key_patches = (assist_slice_number*2 + 1) * (self.feature_size * self.feature_size)
        self.patch_dim = in_channels * self.patch_height * self.patch_width

        self.to_query_patches = nn.Linear(self.patch_dim, self.dmodel)
        self.to_key_patches = nn.Linear(in_channels, self.dmodel)
        self.key_pos_embedding = nn.Parameter(torch.randn(1, self.num_key_patches, self.dmodel), requires_grad=True)
        self.slice_transformer = Transformer_query_response(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, mslice, aslices, class_score, aslices_class):
        # -- firstly: obtain the refer feature for each patch from each aslice, [tensor[B, H, W, n, d]] --
        B, C, H, W = mslice.shape
        mclasses = []
        mclass = class_score
        mclasses.append(mclass)
        class_prob = F.softmax(class_score, dim=1)  # [B C H W]
        mclass_int = torch.argmax(class_prob, dim=1)  # [B H W]
        refer_features = []
        refer_feature, mclass = self.obtain_refer_feature(mclass_int, mslice, class_score, mode="target")
        refer_features.append(refer_feature)
        for i in range(self.assist_slice_number*2):
            refer_feature, mclass = self.obtain_refer_feature(mclass_int, aslices[i], aslices_class[i])
            refer_features.append(refer_feature)
            mclasses.append(mclass)
        mclasses = torch.stack(mclasses, dim=0) # [k b c 32 32]
        mclasses = rearrange(mclasses, 'k b c h w -> b k c h w')
        refer_features = torch.stack(refer_features, dim=0)  # q B H W n d

        # ----------------------------- select the refer slice for each pixel -----------------------------
        class_prob_log = torch.log2(class_prob+0.0001) # B C H W
        #entropy_class =  -1 * torch.sum(mclass * class_prob_log, dim=1) / torch.log2(torch.tensor(self.num_classes*1.0)) # B H W
        entropy_class =  -1 * torch.sum(class_prob * class_prob_log, dim=1) / torch.log2(torch.tensor(self.num_classes*1.0)) # B H W
        confidence = 1-entropy_class # B H W
        #print("min:", torch.min(confidence),"max:", torch.max(confidence))

        # ---------- when the selected range is {1, 3, 5}---------------------- 0~0.4, 0.4~0.7, 0.7~1.0 --------------------------
        if self.assist_slice_number == 2:
            confidence = confidence*0 + 0.1 # 0.1->5, 0.5->3, 0.9->1
            refer_features_selected = refer_features[[0, 0, 0, 0, 0], :, :, :, :, :]
            refer_features_selected[1, confidence<0.8, :, :] = refer_features[2, confidence<0.8, :, :]
            refer_features_selected[2, confidence<0.8, :, :] = refer_features[3, confidence<0.8, :, :]
            refer_features_selected[3, confidence<0.4, :, :] = refer_features[1, confidence<0.4, :, :]
            refer_features_selected[4, confidence<0.4, :, :] = refer_features[4, confidence<0.4, :, :]
        ## ---------- when the selected range is {1, 3, 5, 7}---------------------- 0~0.25, 0.25~0.5, 0.5~0.75, 0.75~1.0-------------- 
        elif self.assist_slice_number == 3:
            confidence = confidence*0 + 0.5 # 0.1->7, 0.5->5, 0.7->3, 0.9->1

            refer_features_selected = refer_features[[0, 0, 0, 0, 0, 0, 0], :, :, :, :, :]
            refer_features_selected[1, confidence<0.8, :, :] = refer_features[3, confidence<0.8, :, :]
            refer_features_selected[2, confidence<0.8, :, :] = refer_features[4, confidence<0.8, :, :]
            refer_features_selected[3, confidence<0.6, :, :] = refer_features[2, confidence<0.6, :, :]
            refer_features_selected[4, confidence<0.6, :, :] = refer_features[5, confidence<0.6, :, :]
            refer_features_selected[5, confidence<0.4, :, :] = refer_features[1, confidence<0.4, :, :]
            refer_features_selected[6, confidence<0.4, :, :] = refer_features[6, confidence<0.4, :, :]
        else:
            confidence = confidence*0 + 0.1 # 0.1->3, 0.9->1
            refer_features_selected = refer_features[[0, 0, 0], :, :, :, :, :]
            refer_features_selected[1, confidence<0.8, :, :] = refer_features[1, confidence<0.8, :, :]
            refer_features_selected[2, confidence<0.8, :, :] = refer_features[2, confidence<0.8, :, :]

        #--------------------------------------------------------------------------------------------------

        refer_features_selected = rearrange(refer_features_selected, 'q B H W n d -> B H W (q n) d')
        # -- secondly: perform the cross-slice transformer --
        query_patches = rearrange(mslice, 'B C (H p1) (W p2) -> (B H W) (p1 p2 C)', p1=self.patch_height, p2=self.patch_width)  # 1024 2014
        query_patches = self.to_query_patches(query_patches[:, None, :])  # BHW 1 D
        key_patches = rearrange(refer_features_selected, 'B H W n d -> (B H W) n d')
        key_patches = self.to_key_patches(key_patches)
        embedded_key_patches = self.key_pos_embedding[:, :self.num_key_patches] + key_patches
        out = self.slice_transformer(query_patches, embedded_key_patches)
        out = rearrange(out, '(B H W) c d -> B (c d) H W', B=B, H=H, W=W)
        return out, mclasses

    def obtain_refer_feature(self, mclass, aslice, aslices_classi, mode="assist"):
        B, H, W = mclass.shape #  (b 16 16)
        aclass0 = aslices_classi * 1
        aclass = torch.argmax(F.softmax(aclass0, dim=1), dim=1)#.detach() # (b 16 16)
        refer_bbox = self.obtain_refer_area(mclass, aclass)  # B H W 4
        refer_roi = rearrange(refer_bbox, 'B H W K -> B (H W) K')
        refer_roi = rearrange(refer_roi, 'B N K -> (B N) K').cuda().float()
        roi_index = torch.zeros((B * H * W))
        for i in range(1, B):
            roi_index[H * W * i: H * W * (i + 1)] = i
        roi_index = roi_index.cuda().float()
        index_roi = torch.cat([roi_index[:, None], refer_roi], dim=1)
        xy_index_roi = index_roi[:, [0, 2, 1, 4, 3]].contiguous()
        refer_feature = roi_align(aslice, xy_index_roi, (self.feature_size, self.feature_size))  # BN d h w
        refer_feature = rearrange(refer_feature, '(B N) d h w -> B N (h w) d', B=B)  # B N n d
        refer_feature = rearrange(refer_feature, 'B (H W) n d -> B H W n d', H=H)  # B H W n d
        return refer_feature, aclass0

    def obtain_refer_area(self, mclass, aclass, mode="assist"):
        # firstly, the box over each patch is the refer area of the corresponding patch
        b, h, w = mclass.shape
        init_bbox = self.initial_refer_area(mclass).repeat(b, 1, 1, 1)  # b h w 4 (b, 16, 16, 4)
        # secondly obtain the foreground bbox of the refer slice (assist slice)
        for i in range(b):
            foreground_bbox_i = self.obtain_foreground_bbox(aclass[i, :, :])
        # thirdly, update the refer area of foreground to the responding foreground
            init_bbox_i = init_bbox[i, :, :, :]  # h w 4
            for j in range(1, self.num_classes):
                if foreground_bbox_i[j, 0] == 1:
                    init_bbox_i[mclass[i] == j, :] = foreground_bbox_i[j, 1:]
            init_bbox[i, :, :, :] = init_bbox_i
        return init_bbox

    def initial_refer_area(self, mclass):
        factor = 1
        b, h, w = mclass.shape
        x_vec = np.arange(0, w)
        y_vec = np.arange(0, h)
        xv, yv = np.meshgrid(x_vec, y_vec)
        init_bbox = np.zeros((1, h, w, 4))  # y1, x1, y2, x2
        init_bbox[0, :, :, 0] = factor * yv - self.init_size // 2
        init_bbox[0, :, :, 1] = factor * xv - self.init_size // 2
        init_bbox[0, :, :, 2] = factor * yv + self.init_size // 2
        init_bbox[0, :, :, 3] = factor * xv + self.init_size // 2
        init_bbox = init_bbox.clip(0, factor * h - 1)
        return torch.tensor(init_bbox).float()

    def obtain_foreground_bbox(self, aclass_i, mode="assist", buffer=0.2):
        foreground_bbox = torch.zeros((self.num_classes, 5)) # flag, y1, x1, y2, x2
        for i in range(1, self.num_classes):
            a = torch.where(aclass_i == i)
            #print(len(a[0]))
            if len(a[0]) > 0:
                #print("yes")
                foreground_bbox[i, 0] = 1
                foreground_bbox[i, 1] = torch.min(a[0]).item()
                foreground_bbox[i, 3] = torch.max(a[0]).item()
                foreground_bbox[i, 2] = torch.min(a[1]).item()
                foreground_bbox[i, 4] = torch.max(a[1]).item()
        # Extend the useful information scope appropriately
        width = foreground_bbox[:, 4] - foreground_bbox[:, 2]
        height = foreground_bbox[:, 3] - foreground_bbox[:, 1]
        foreground_bbox[:, 1] = foreground_bbox[:, 1] - torch.ceil(buffer*height)
        foreground_bbox[:, 2] = foreground_bbox[:, 2] - torch.ceil(buffer*width)
        foreground_bbox[:, 3] = foreground_bbox[:, 3] + torch.ceil(buffer*height)
        foreground_bbox[:, 4] = foreground_bbox[:, 4] + torch.ceil(buffer*width)
        foreground_bbox = foreground_bbox.clamp(0, aclass_i.shape[0]-1)
        if mode == "notarget":
            visual_refer_region(foreground_bbox[:, 1:])
        return foreground_bbox
    