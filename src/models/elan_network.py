import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.elan_block import ELAB, MeanShift

def create_model(args):
    return ELAN(args)

class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.scale, self.scale, w // self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale ** 2), h // self.scale, w // self.scale)
        return x

class ELAN(nn.Module):
    def __init__(self, args):
        super(ELAN, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.m_elan  = args.m_elan
        self.c_elan  = args.c_elan
        self.n_share = args.n_share
        self.r_expand = args.r_expand
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        if self.scale != 4:
            self.enlarge_factor = self.scale

        else:
            self.enlarge_factor = self.scale // 2 # 2
            
        m_head1 = [
            DownBlock(self.enlarge_factor),
            nn.Conv2d(self.colors * ( ( self.enlarge_factor) ** 2), self.c_elan // 2, 3, 1, 1),
        ]
        m_head2 = [
            DownBlock(self.enlarge_factor),
            nn.Conv2d(self.colors * ( ( self.enlarge_factor) ** 2), self.c_elan - self.c_elan // 2, 5, 1, 2),
        ]
        head_merge = [
            nn.Conv2d(self.c_elan, self.c_elan, 1, 1, 0)
        ]
        
        

        # define head module
        #m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)] # 2 * 9 * 3 * 48 * (32 * 32 or 21 * 21 or 16 * 16)

        # define body module
        
        '''
        m_body = []
        
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 0, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 1, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        '''
        
        '''
        self.m_body = []
        for i in range(self.m_elan // 2):
            buf = [
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
            ]
            buf = nn.Sequential(*buf)

            self.m_body.append(buf)
        '''
        
        self.m_body0 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])
        self.m_body1 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])
        
        self.m_body2 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])
        self.m_body3 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])

        self.m_body4 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])
        self.m_body5 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])

        self.m_body6 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])
        self.m_body7 = nn.Sequential( *[
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 0, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
                ELAB(
                    self.c_elan, self.c_elan, self.r_expand, 1, 
                    self.window_sizes, shared_depth=self.n_share
                    ),
        ])
        '''
        self.shrink_channel = []
        for i in range(self.m_elan // 4 - 1):
            self.shrink_channel.append(nn.Conv2d((i + 2) * self.c_elan, self.c_elan, kernel_size = 3, stride = 1, padding = 1))
        '''
        self.shrink_channel0 = nn.Conv2d(2 * self.c_elan, self.c_elan, kernel_size = 3, stride = 1, padding = 1)
        self.shrink_channel1 = nn.Conv2d(3 * self.c_elan, self.c_elan, kernel_size = 3, stride = 1, padding = 1)
        self.shrink_channel2 = nn.Conv2d(4 * self.c_elan, self.c_elan, kernel_size = 3, stride = 1, padding = 1)
        '''
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors*self.scale*self.scale, kernel_size=3, stride=1, padding=1), # 2 * 9 * 48 * 3 * s2 * (32 * 32 or 21 * 21 or 16 * 16)
            nn.PixelShuffle(self.scale)
        ]
        '''
        # define tail module
        # 
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors*(self.enlarge_factor)*(self.enlarge_factor), kernel_size=3, stride=1, padding=1), # 2 * 9 * 48 * 3 * s2 * (32 * 32 or 21 * 21 or 16 * 16)
            nn.PixelShuffle(self.enlarge_factor)
        ]

        self.head1 = nn.Sequential(*m_head1)
        self.head2 = nn.Sequential(*m_head2)
        self.head_merge = nn.Sequential(*head_merge)
        #self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        x = torch.concat((x1, x2), dim = 1)
        x = self.head_merge(x)
        H, W = x.shape[2:]
        #up_img = F.interpolate(x, scale_factor = self.scale, mode = 'bicubic')
        #print(f'before, H = {H}, W = {W}, x.shape = {x.shape}')
        x = self.check_image_size(x)
        #print(f'after, x.shape = {x.shape}')        
        
        
        #res = self.body(x)
        
        cur = x
        res = cur
        shortcut = res
        res = self.m_body0(res) + shortcut
        shortcut = res
        res = self.m_body1(res) + shortcut
        #x = res

        cur = torch.cat((cur, res), dim = 1)
        res = self.shrink_channel0(cur)
        shortcut = res
        res = self.m_body2(res) + shortcut
        shortcut = res
        res = self.m_body3(res) + shortcut
        #x = res

        cur = torch.cat((cur, res), dim = 1)
        res = self.shrink_channel1(cur)
        shortcut = res
        res = self.m_body4(res) + shortcut
        shortcut = res
        res = self.m_body5(res) + shortcut
        #x = res

        cur = torch.cat((cur, res), dim = 1)
        res = self.shrink_channel2(cur)
        shortcut = res
        res = self.m_body6(res) + shortcut
        shortcut = res
        res = self.m_body7(res) + shortcut
        #x = res

        
        
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)
        
        #print(f'H = {H}, W = {W}, x.shape = {x[:, :, 0:H*self.scale, 0:W*self.scale].shape}')
        #return x[:, :, 0:H*(self.scale), 0:W*(self.scale)]# + up_img 
        return x[:, :, 0:H*(self.enlarge_factor), 0:W*(self.enlarge_factor)]# + up_img # For x4 only
        #return x[:, :, 0 : H, 0 : W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    pass