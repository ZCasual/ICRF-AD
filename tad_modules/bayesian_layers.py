import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GaussianVariational(nn.Module):
    """高斯变分分布 - 贝叶斯权重的后验分布实现"""
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps', None)
        self.sigma = None
        
    def sample(self):
        """从分布中采样"""
        if self.eps is None or self.eps.shape != self.mu.shape:
            self.eps = torch.randn_like(self.mu)
        
        self.sigma = torch.log1p(torch.exp(self.rho))
        return self.mu + self.sigma * self.eps
    
    def log_prob(self, x):
        """计算对数概率密度"""
        if self.sigma is None:
            self.sigma = torch.log1p(torch.exp(self.rho))
            
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixture(nn.Module):
    """尺度混合高斯分布 - 用于权重先验"""
    def __init__(self, pi=0.5, sigma1=1.0, sigma2=0.002):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
    def log_prob(self, x):
        """计算对数概率密度"""
        prob1 = torch.exp(-x.pow(2) / (2 * self.sigma1 ** 2)) / (math.sqrt(2 * math.pi) * self.sigma1)
        prob2 = torch.exp(-x.pow(2) / (2 * self.sigma2 ** 2)) / (math.sqrt(2 * math.pi) * self.sigma2)
        return torch.log(self.pi * prob1 + (1 - self.pi) * prob2).sum()

class BayesianConv2d(nn.Module):
    """贝叶斯卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        
        # 权重变分后验
        weight_mu = torch.Tensor(out_channels, in_channels, *self.kernel_size)
        weight_rho = torch.Tensor(out_channels, in_channels, *self.kernel_size)
        
        # 初始化参数
        nn.init.kaiming_uniform_(weight_mu, a=math.sqrt(5))
        nn.init.constant_(weight_rho, -3)  # 初始方差很小
        
        self.weight = GaussianVariational(weight_mu, weight_rho)
        
        # 偏置变分后验
        if bias:
            bias_mu = torch.Tensor(out_channels)
            bias_rho = torch.Tensor(out_channels)
            nn.init.constant_(bias_mu, 0.0)
            nn.init.constant_(bias_rho, -3)
            self.bias = GaussianVariational(bias_mu, bias_rho)
        else:
            self.register_parameter('bias', None)
            
        # 权重先验
        self.weight_prior = ScaleMixture()
        if bias:
            self.bias_prior = ScaleMixture()
            
        # KL散度
        self.kl_divergence = 0.0
        
    def forward(self, x):
        """前向传播，包括KL散度计算"""
        weight = self.weight.sample()
        
        if self.use_bias:
            bias = self.bias.sample()
        else:
            bias = None
            
        # 计算KL散度
        self.kl_divergence = self.weight.log_prob(weight) - self.weight_prior.log_prob(weight)
        if self.use_bias:
            self.kl_divergence += self.bias.log_prob(bias) - self.bias_prior.log_prob(bias)
        
        # 执行卷积
        out = F.conv2d(
            x, weight, bias, 
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return out

class BayesianConvBlock(nn.Module):
    """贝叶斯卷积块，包括卷积、归一化和激活"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = BayesianConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
        
    @property
    def kl_divergence(self):
        """返回块的KL散度"""
        return self.conv.kl_divergence 