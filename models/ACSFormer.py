from .components.unets_parts import *
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from .components.acsformer_parts import Intra_slice_Classaware_Interaction,  Inter_slice_Adaptive_Interaction

class ClassForward(nn.Module):
    def __init__(self, dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class ACSFormer(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ACSFormer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_dim = 48
        self.assist_slice_number = 3

        self.inc = DoubleConv(n_channels, self.base_dim)
        self.down1 = Down(self.base_dim, self.base_dim*2)
        self.down2 = Down(self.base_dim*2, self.base_dim*4)
        self.down3 = Down(self.base_dim*4, self.base_dim*8)
        self.down4 = Down(self.base_dim*8, self.base_dim*16)

        self.to_class4 =  nn.Conv2d(self.base_dim*8, n_classes, kernel_size=1, bias=False)
        self.to_class5 =  nn.Conv2d(self.base_dim*16, n_classes, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.icgi = Intra_slice_Classaware_Interaction(n_channels=self.base_dim*8, n_classes=n_classes, n_space=4)
        self.intra_class = Inter_slice_Adaptive_Interaction(self.base_dim*16, self.base_dim*16, n_classes, depth=1, patch_size=1, assist_slice_number=self.assist_slice_number)

        self.up1 = Up(self.base_dim*16, self.base_dim*8, bilinear)
        self.up2 = Up(self.base_dim*8, self.base_dim*4, bilinear)
        self.up3 = Up(self.base_dim*4, self.base_dim*2, bilinear)
        self.up4 = Up(self.base_dim*2, self.base_dim, bilinear)
        self.outc = OutConv(self.base_dim, n_classes)

        #-----------------------------------------select the slice spacing-----------------------------------------
        if self.assist_slice_number == 2:
            self.group = torch.tensor([[5, 6, 7, 8], [3, 5, 8, 10], [1, 4, 9, 12]]).cuda() # for the selection range of {1,3,5}
        elif self.assist_slice_number == 3:
            self.group = torch.tensor([[7, 8, 9, 10, 11, 12], [4, 6, 8, 11, 13, 15], [1, 4, 7, 12, 15, 18]]).cuda() # for the selection range of {1,3,5,7}
        else:
            self.group = torch.tensor([[3, 4], [2, 5], [1, 6]]).cuda()  # iter = 1, 2, 3

    def forward(self, slices):
        b, z, c, H, W = slices.shape
        if H==512:
            slices = rearrange(slices, 'b z c h w -> (b z) c h w')
            slices = F.interpolate(slices, (256, 256), mode="bilinear", align_corners=False)
            slices = rearrange(slices, '(b z) c h w -> b z c h w', z=z)


        x1 = self.inc(slices[:, 0, :, :, :])
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        class_score0 = self.to_class4(x4)
        x4 = self.icgi(x4, class_score0)

        # ------------ select the slice spacing ------------
        class_score = self.to_class5(x5)
        with torch.no_grad():
            b, c, h, w = class_score.shape
            aslices = []
            aslices_class = []
            class_prob = self.softmax(class_score).reshape(b, c, -1) # [b c N]
            class_prob_log = torch.log2(class_prob+0.0001) # [b c N]
            entropy_class =  -1 * torch.sum(class_prob * class_prob_log, dim=1) / torch.log2(torch.tensor(self.n_classes*1.0)) # [b N]
            #print("minconf:", 1-torch.max(entropy_class, dim=-1)[0])
            #confidence = 1-torch.mean(entropy_class, dim=-1) # [b]
            confidence = 1-torch.max(entropy_class, dim=-1)[0] 
            if self.assist_slice_number == 2:
                selected_spacing_slices = slices[:, 5:9, :, :, :] # when the selection range of {1, 3, 5}
            elif self.assist_slice_number == 3:
                selected_spacing_slices = slices[:, 7:13, :, :, :]
            else:
                selected_spacing_slices = slices[:, 3:5, :, :, :]
            #print("maxconf:", confidence)
            for i in range(b):
                if confidence[i] >=0.9:  #0.9999
                    selected_spacing_slices[i, :, :, :, :] = slices[i, self.group[2], :, :, :]
                else:
                    if confidence[i] >= 0.6:  #0.7
                        selected_spacing_slices[i, :, :, :, :] = slices[i, self.group[1], :, :, :]
                #selected_spacing_slices[i, :, :, :, :] = slices[i, self.group[2], :, :, :]

            for i in range(self.assist_slice_number*2):
                slicei = self.inc(selected_spacing_slices[:, i, :, :, :])
                slicei = self.down1(slicei)
                slicei = self.down2(slicei)
                slicei = self.down3(slicei)
                slicei = self.down4(slicei)
                slicei_class = self.to_class5(slicei)
                aslices.append(slicei)
                aslices_class.append(slicei_class)
        intra, mclasses = self.intra_class(x5, aslices, class_score, aslices_class)
        x5 = x5 + intra

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if H==512:
            logits = F.interpolate(logits, (512, 512), mode="bilinear", align_corners=False)
            class_score = F.interpolate(class_score, (512//16, 512//16), mode="bilinear", align_corners=False)
            class_score0 = F.interpolate(class_score0, (512//8, 512//8), mode="bilinear", align_corners=False)
        return logits, class_score, class_score0





