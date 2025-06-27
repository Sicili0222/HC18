class mamba_block(nn.Module):
    def __init__(self, in_ch):
        super(mamba_block, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduction = 4 
        mid_channels = max(16, in_ch // reduction) 
        num_groups = min(8, in_ch)
        while in_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_ch, mid_channels, 1, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_ch, 1, bias=False)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.GroupNorm(1, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.GroupNorm(num_groups, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1, bias=False)
        )
        
        self.gate = nn.Parameter(torch.zeros(1) + 0.5)
        self.norm = nn.GroupNorm(num_groups, in_ch) 

        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=i, dilation=i),
                nn.GroupNorm(num_groups, in_ch),  
                nn.ReLU(inplace=True)
            ) for i in [1, 2, 3]
        ])

    def forward(self, x):
        identity = x

        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.spatial_conv(spatial_feat)

        multi_scale_feats = []
        for conv in self.multi_scale:
            multi_scale_feats.append(conv(x))
        multi_scale_out = sum(multi_scale_feats) / len(self.multi_scale)
        
        edge_feat = self.edge_enhance(x)
        
        out = x * channel_att * self.gate + \
              x * spatial_att * (1 - self.gate) + \
              edge_feat * 0.2 + \
              multi_scale_out * 0.1
              
        out = self.norm(out)
        return out + identity

class res_conv(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, mamba_level='default'):
        super(res_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if mamba_level == 'default':
            self.use_mamba = out_ch >= 128
        elif mamba_level == 'all':  
            self.use_mamba = True
        elif mamba_level == 'none':
            self.use_mamba = False
        elif mamba_level == 'encoder_only': 
            self.use_mamba = out_ch >= 128 and down
        elif mamba_level == 'decoder_only': 
            self.use_mamba = out_ch >= 128 and not down
        elif mamba_level == 'bottleneck_only':  
            self.use_mamba = out_ch == 512
        elif mamba_level == 'shallow_only':
            self.use_mamba = out_ch <= 128
        
        if self.use_mamba:
            self.mamba = mamba_block(out_ch)
        
        # self.cbam=CBAM(out_ch)
        # self.bfam=BFAM(out_ch,out_ch)
        # self.cdfa=CDFAPreprocess(out_ch,out_ch,1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1
        if self.use_mamba:
            x2 = self.mamba(x2) 
            # x2=self.cbam(x2)
            # x2=self.bfam(x2,x2)
            # x2=self.cdfa(x2)
        return x2

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, mamba_level='default'):
        super(inconv, self).__init__()
        self.conv = res_conv(in_ch, out_ch, mamba_level=mamba_level)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, mamba_level='default'):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            res_conv(in_ch, out_ch, down=True, mamba_level=mamba_level),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, mamba_level='default'):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = res_conv(in_ch, out_ch, down=False, mamba_level=mamba_level)
        
        if mamba_level == 'default':
            self.use_mamba = out_ch >= 128
        elif mamba_level == 'all':
            self.use_mamba = True
        elif mamba_level == 'none':
            self.use_mamba = False
        elif mamba_level == 'encoder_only':
            self.use_mamba = False  # 上采样属于decoder
        elif mamba_level == 'decoder_only':
            self.use_mamba = out_ch >= 128
        elif mamba_level == 'bottleneck_only':
            self.use_mamba = out_ch == 512
        elif mamba_level == 'shallow_only':
            self.use_mamba = out_ch <= 128
            
        if self.use_mamba:
            self.mamba = mamba_block(out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff1 = x2.shape[2] - x1.shape[2]
        diff2 = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, pad=(diff1 // 2, diff1 - diff1 // 2, diff2 // 2, diff2 - diff2 // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_mamba:
            x = self.mamba(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, mamba_level='default'):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        
        if mamba_level == 'default':
            self.use_mamba = out_ch > 1
        elif mamba_level == 'all':
            self.use_mamba = True
        elif mamba_level == 'none':
            self.use_mamba = False
        elif mamba_level == 'encoder_only':
            self.use_mamba = False
        elif mamba_level == 'decoder_only':
            self.use_mamba = True
        elif mamba_level == 'bottleneck_only':
            self.use_mamba = False
        elif mamba_level == 'shallow_only':
            self.use_mamba = True
            
        if self.use_mamba:
            self.mamba = mamba_block(out_ch)

    def forward(self, x):
        x = self.conv(x)
        if self.use_mamba:
            x = self.mamba(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, mamba_level='default'):
        super(ResUNet, self).__init__()
        self.inc = inconv(n_channels, 64, mamba_level=mamba_level)
        self.down1 = down(64, 128, mamba_level=mamba_level)
        self.down2 = down(128, 256, mamba_level=mamba_level)
        self.down3 = down(256, 512, mamba_level=mamba_level)
        self.down4 = down(512, 512, mamba_level=mamba_level)
        self.up1 = up(1024, 256, mamba_level=mamba_level)
        self.up2 = up(512, 128, mamba_level=mamba_level)
        self.up3 = up(256, 64, mamba_level=mamba_level)
        self.up4 = up(128, 64, mamba_level=mamba_level)
        self.outc = outconv(64, n_classes, mamba_level=mamba_level)
        self.dropout = torch.nn.Dropout2d(0.3)

        self.edge_refinement = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1)
        )

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
      
        main_out = self.outc(x)
      
        edge_out = self.edge_refinement(x)

        combined_out = main_out + 0.1 * edge_out
        sigmoid_out = torch.sigmoid(combined_out)
        
        return sigmoid_out


model = ResUNet(1, 1)
model.to('cuda')
print("Model Loaded to GPU")
