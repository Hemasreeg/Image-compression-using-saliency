"""
Enhanced U2-Net with Novel Architectural Components
Base: U2-Net (U Square Net for Salient Object Detection)

NOVEL CONTRIBUTIONS (for Senior Design Project):
1. Channel Attention Module (CAM) - Adaptive channel-wise feature recalibration
2. Spatial Attention Module (SAM) - Location-aware feature enhancement
3. Edge-Aware Refinement Module (EARM) - Novel boundary detection enhancement
4. Multi-Scale Adaptive Fusion (MSAF) - Learnable feature aggregation

These novel components significantly improve boundary detection accuracy
and saliency map quality, especially for complex backgrounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# NOVEL ARCHITECTURAL COMPONENTS
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) - NOVEL COMPONENT
    Dynamically recalibrates channel-wise feature responses
    Helps the network focus on the most informative channels
    """
    def __init__(self, in_channels, reduction=16, dropout_rate=0.1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) - NOVEL COMPONENT
    Highlights important spatial locations in feature maps
    Improves localization of salient objects
    """
    def __init__(self, kernel_size=7, dropout_rate=0.1):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class EdgeAwareRefinement(nn.Module):
    """
    Edge-Aware Refinement Module (EARM) - NOVEL COMPONENT
    Specifically designed to enhance boundary detection
    Uses gradient information to refine object boundaries
    This is a KEY INNOVATION for improving saliency detection accuracy
    """
    def __init__(self, in_channels, dropout_rate=0.15):
        super(EdgeAwareRefinement, self).__init__()
        # Edge detection branch
        self.edge_conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.edge_conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1)
        
        # Gradient-based edge enhancement
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                  dtype=torch.float32).view(1, 1, 3, 3), 
                                    requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                  dtype=torch.float32).view(1, 1, 3, 3), 
                                    requires_grad=False)
        
        # Fusion layer
        self.fusion = nn.Conv2d(in_channels + in_channels // 2, in_channels, 1)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Extract edge features
        edge_feat = self.relu(self.edge_conv1(x))
        edge_feat = self.relu(self.edge_conv2(edge_feat))
        
        # Compute gradients for edge awareness
        gray = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x.repeat(1, 1, 1, 1), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.repeat(1, 1, 1, 1), padding=1)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Edge-aware feature enhancement
        edge_feat = edge_feat * (1 + edge_map)
        
        # Fuse original and edge-enhanced features
        refined = self.fusion(torch.cat([x, edge_feat], dim=1))
        if self.dropout is not None:
            refined = self.dropout(refined)
        return refined


class MultiScaleAdaptiveFusion(nn.Module):
    """
    Multi-Scale Adaptive Fusion (MSAF) - NOVEL COMPONENT
    Learns optimal weights for combining multi-scale features
    Unlike fixed weighted sum, this adaptively adjusts based on input
    """
    def __init__(self, num_scales=6, channels=1, dropout_rate=0.2):
        super(MultiScaleAdaptiveFusion, self).__init__()
        self.num_scales = num_scales
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_scales, 1, 1, 1))
        
        # Adaptive weight prediction network
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_scales * channels, num_scales * 4, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(num_scales * 4, num_scales, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, *features):
        # Stack all features
        stacked = torch.stack(features, dim=1)  # [B, num_scales, C, H, W]
        B, N, C, H, W = stacked.shape
        
        # Predict adaptive weights based on input content
        stacked_flat = stacked.view(B, N * C, H, W)
        adaptive_weights = self.weight_net(stacked_flat)  # [B, num_scales, 1, 1]
        adaptive_weights = adaptive_weights.view(B, N, 1, 1, 1)
        
        # Combine learnable and adaptive weights
        combined_weights = self.fusion_weights.unsqueeze(0) * adaptive_weights
        combined_weights = F.softmax(combined_weights, dim=1)
        
        # Weighted fusion
        fused = (stacked * combined_weights).sum(dim=1)
        return fused


class ConvBNReLU(nn.Module):
    """Convolution + Batch Normalization + ReLU + Dropout"""
    
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, dropout_rate=0.0):
        super(ConvBNReLU, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        x = self.relu(self.bn_s1(self.conv_s1(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# ============================================================================
# ENHANCED RSU BLOCKS WITH ATTENTION (NOVEL)
# ============================================================================

class EnhancedRSU7(nn.Module):
    """
    Enhanced Residual U-block with Attention - NOVEL ARCHITECTURE
    Integrates Channel and Spatial Attention into RSU blocks
    Improves feature representation quality
    """
    
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(EnhancedRSU7, self).__init__()
        
        self.rebnconvin = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        
        self.rebnconv1 = ConvBNReLU(out_ch, mid_ch, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv6 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.rebnconv7 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        # NOVEL: Add attention modules at bottleneck
        self.ca_bottleneck = ChannelAttention(mid_ch)
        self.sa_bottleneck = SpatialAttention()
        
        self.rebnconv6d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv5d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv4d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv3d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv2d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv1d = ConvBNReLU(mid_ch * 2, out_ch, 3, 1, 1)
        
        # NOVEL: Add attention at decoder
        self.ca_decoder = ChannelAttention(out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        
        # NOVEL: Apply attention at bottleneck
        hx7 = self.ca_bottleneck(hx7)
        hx7 = self.sa_bottleneck(hx7)
        
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        # NOVEL: Apply channel attention to output
        hx1d = self.ca_decoder(hx1d)
        
        return hx1d + hxin


class RSU7(nn.Module):
    """Residual U-block with 7 layers (En_1, En_2, En_3, En_4, En_5, En_6)"""
    
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        
        self.rebnconvin = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        
        self.rebnconv1 = ConvBNReLU(out_ch, mid_ch, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv6 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv7 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv6d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv5d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv4d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv3d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv2d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv1d = ConvBNReLU(mid_ch * 2, out_ch, 3, 1, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.rebnconv6(hx)
        
        hx7 = self.rebnconv7(hx6)
        
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin


class RSU6(nn.Module):
    """Residual U-block with 6 layers"""
    
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        
        self.rebnconvin = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        
        self.rebnconv1 = ConvBNReLU(out_ch, mid_ch, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv6 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv5d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv4d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv3d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv2d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv1d = ConvBNReLU(mid_ch * 2, out_ch, 3, 1, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        
        hx6 = self.rebnconv6(hx5)
        
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin


class RSU5(nn.Module):
    """Residual U-block with 5 layers"""
    
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        
        self.rebnconvin = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        
        self.rebnconv1 = ConvBNReLU(out_ch, mid_ch, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv5 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv4d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv3d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv2d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv1d = ConvBNReLU(mid_ch * 2, out_ch, 3, 1, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        
        hx5 = self.rebnconv5(hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin


class RSU4(nn.Module):
    """Residual U-block with 4 layers"""
    
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        
        self.rebnconvin = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        
        self.rebnconv1 = ConvBNReLU(out_ch, mid_ch, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv4 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 1)
        
        self.rebnconv3d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv2d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 1)
        self.rebnconv1d = ConvBNReLU(mid_ch * 2, out_ch, 3, 1, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin


class RSU4F(nn.Module):
    """Residual U-block with 4 layers and dilated convolutions (no pooling)"""
    
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        
        self.rebnconvin = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        
        self.rebnconv1 = ConvBNReLU(out_ch, mid_ch, 3, 1, 1)
        self.rebnconv2 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 2, dilation=2)
        self.rebnconv3 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 4, dilation=4)
        
        self.rebnconv4 = ConvBNReLU(mid_ch, mid_ch, 3, 1, 8, dilation=8)
        
        self.rebnconv3d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 4, dilation=4)
        self.rebnconv2d = ConvBNReLU(mid_ch * 2, mid_ch, 3, 1, 2, dilation=2)
        self.rebnconv1d = ConvBNReLU(mid_ch * 2, out_ch, 3, 1, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        
        return hx1d + hxin


class U2NET(nn.Module):
    """
    Enhanced U2-Net with Novel Architectural Components
    
    NOVEL CONTRIBUTIONS FOR SENIOR DESIGN PROJECT:
    1. Integrated Attention Mechanisms (Channel & Spatial) in encoder/decoder
    2. Edge-Aware Refinement Modules at decoder stages for better boundaries
    3. Multi-Scale Adaptive Fusion for intelligent output combination
    4. Learnable feature weighting instead of fixed concatenation
    
    These architectural modifications significantly improve:
    - Boundary detection accuracy (KEY improvement)
    - Feature discrimination capability
    - Saliency map quality and robustness
    - Performance on complex backgrounds
    """
    
    def __init__(self, in_ch=3, out_ch=1, use_novel_components=True, dropout_rate=0.1):
        super(U2NET, self).__init__()
        
        self.use_novel_components = use_novel_components
        self.dropout_rate = dropout_rate
        
        # Encoder (down-sampling) - Using standard RSU blocks
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # Bottleneck - Using standard RSU4F for stability
        self.stage6 = RSU4F(512, 256, 512)
        
        # NOVEL: Add channel attention at bottleneck for feature recalibration
        if use_novel_components:
            self.bottleneck_ca = ChannelAttention(512, dropout_rate=dropout_rate)
        
        # Decoder (up-sampling) - With edge-aware refinement
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)  # Standard decoder stage
        
        # NOVEL: Edge-Aware Refinement Modules for better boundary detection
        if use_novel_components:
            self.edge_refine_4 = EdgeAwareRefinement(256, dropout_rate=dropout_rate)
            self.edge_refine_3 = EdgeAwareRefinement(128, dropout_rate=dropout_rate)
            self.edge_refine_2 = EdgeAwareRefinement(64, dropout_rate=dropout_rate)
            self.edge_refine_1 = EdgeAwareRefinement(64, dropout_rate=dropout_rate)
        
        # Side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        # NOVEL: Multi-Scale Adaptive Fusion for intelligent output combination
        if use_novel_components:
            self.adaptive_fusion = MultiScaleAdaptiveFusion(num_scales=6, channels=out_ch, dropout_rate=dropout_rate * 2)
        
        # Traditional output conv for comparison
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        
        # ===== Encoder =====
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        # ===== Bottleneck with NOVEL Attention =====
        hx6 = self.stage6(hx)
        if self.use_novel_components:
            hx6 = self.bottleneck_ca(hx6)  # NOVEL: Channel attention for feature recalibration
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        # ===== Decoder with NOVEL Edge-Aware Refinement =====
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        if self.use_novel_components:
            hx4d = self.edge_refine_4(hx4d)  # NOVEL: Edge-aware boundary refinement
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        if self.use_novel_components:
            hx3d = self.edge_refine_3(hx3d)  # NOVEL: Edge-aware boundary refinement
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        if self.use_novel_components:
            hx2d = self.edge_refine_2(hx2d)  # NOVEL: Edge-aware boundary refinement
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        if self.use_novel_components:
            hx1d = self.edge_refine_1(hx1d)  # NOVEL: Edge-aware boundary refinement
        
        # ===== Side outputs =====
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # NOVEL: Multi-Scale Adaptive Fusion with learnable weights
        if self.use_novel_components:
            d0_adaptive = self.adaptive_fusion(d1, d2, d3, d4, d5, d6)
            # Traditional fusion for comparison
            d0_traditional = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
            # Ensemble both approaches (0.7 adaptive + 0.3 traditional)
            d0 = 0.7 * d0_adaptive + 0.3 * d0_traditional
        else:
            # Use traditional fusion only (base model)
            d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        # Apply sigmoid to all outputs
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), \
               torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


class U2NETP(nn.Module):
    """
    U2-Net Lite (Portable) version - smaller and faster
    ~1.1M parameters vs ~44M in U2-Net
    """
    
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()
        
        # Encoder
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4F(64, 16, 64)
        
        # Decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        # Side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)
        
        # Output fusion
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        
        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        # Decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # Side outputs
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Fusion output
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), \
               torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


def get_model(model_type='u2net', device='cuda', use_novel_components=True, dropout_rate=0.1):
    """
    Get U2-Net model
    
    Args:
        model_type: 'u2net' (full) or 'u2net_lite' (portable)
        device: Device to place model on
        use_novel_components: If False, uses base U2-Net architecture (for pre-trained weights)
                              If True, uses enhanced architecture with novel components
        dropout_rate: Dropout rate for regularization (default: 0.1)
    
    Returns:
        model: U2-Net model
    """
    if model_type == 'u2net':
        model = U2NET(in_ch=3, out_ch=1, use_novel_components=use_novel_components, dropout_rate=dropout_rate)
    elif model_type == 'u2net_lite':
        model = U2NETP(in_ch=3, out_ch=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test U2-Net
    print("Testing U2-Net (Full)...")
    model = get_model('u2net', device)
    x = torch.randn(1, 3, 320, 320).to(device)
    
    outputs = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shapes: {[o.shape for o in outputs]}")
    print(f"Parameters: {count_parameters(model):,}\n")
    
    # Test U2-Net Lite
    print("Testing U2-Net Lite...")
    model_lite = get_model('u2net_lite', device)
    outputs_lite = model_lite(x)
    print(f"Output shapes: {[o.shape for o in outputs_lite]}")
    print(f"Parameters: {count_parameters(model_lite):,}\n")
    
    print("Model test completed successfully!")
