import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import time
import random
class LightweightAttention(nn.Module):
    def __init__(self, dim, num_heads=4, drop_rate=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        
        self.proj = nn.Linear(dim, dim*3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.pos_emb = nn.Parameter(torch.randn(1, 1, dim))
        self.drop_rate = drop_rate

    def forward(self, x):
        bhw, n, f = x.shape
        x = x + self.pos_emb  # add positional embedding
        
        # generate queries, keys, values
        qkv = self.proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'bhw n (h d) -> bhw h n d', h=self.num_heads), qkv)# h * d = f
        # calculate attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        # print("1st head attn matrix in first pixel",attn[0,0,:,:])
        """
        add when test robustness
        """

        ########################################################################
        out = torch.matmul(attn, v)

        out = rearrange(out, 'bhw h n d -> bhw n (h d)')  
        # result = (self.out_proj(out)+x).mean(dim=1) if n<=2 else (self.out_proj(out)+x_reshaped).sum(dim=1)/(n - 1)
        result= (self.out_proj(out)+x).mean(dim=1)
        return result  # average over heads

    def drop_and_reweight(self, attn_weights):
        bhw, h, q, k = attn_weights.shape
        
        min_indices = attn_weights.mean(dim=1).argmin(dim=-1)
        
        indices = min_indices.view(bhw, 1, q, 1).expand(-1, h, -1, -1)

        mask = torch.ones_like(attn_weights, device=attn_weights.device)
        mask.scatter_(dim=3, index=indices, value=0)  
    
        masked_attn = attn_weights * mask
        return masked_attn / (masked_attn.sum(dim=-1, keepdim=True) + 1e-6),indices
    

class LightweightAttention_singlehead(nn.Module):
    def __init__(self, dim, drop_rate=0.1):
        super().__init__()
        self.scale = dim ** -0.5
        
        self.proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.pos_emb = nn.Parameter(torch.randn(1, 1, dim))
        self.drop_rate = drop_rate

    def forward(self, x):
        bhw, n, f = x.shape
        x = x + self.pos_emb  # add positional embedding
        
        # generate queries, keys, values
        qkv = self.proj(x).chunk(3, dim=-1)  # q, k, v: (bhw, n, dim)
        q, k, v = qkv
        
        # attention: (bhw, n, n)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print("attn matrix in first pixel",attn[0,:,:])
        # out: (bhw, n, dim)
        out = torch.matmul(attn, v)
        # print("out matrix in first pixel",out[0,:,:])
        projected_out = self.out_proj(out)
        # print(x[0,:,:])
        # print("projected_out matrix in first pixel",projected_out[0,:,:])
        result = (projected_out + x).mean(dim=1)
        return result
    
class LightweightAttention_singlehead_drop(LightweightAttention_singlehead):
    def forward(self, x, threshold=0.01):
        bhw, n, f = x.shape
        x = x + self.pos_emb  # add positional embedding
        
        # generate queries, keys, values
        qkv = self.proj(x).chunk(3, dim=-1)  # q, k, v: (bhw, n, dim)
        q, k, v = qkv
        # attention: (bhw, n, n)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print("attn matrix in temp pixel",attn[0,:,:])
        attn = self.drop_and_reweight_compare_diagonal_threshold(attn, threshold=threshold)
        # print("1st head reweight attn matrix in first pixel",attn[0,:,:])
        out = torch.matmul(attn, v)
        # print("out matrix in first pixel",out[0,:,:])
        keep_rows = out.sum(dim=-1)>0
        # keep_mask = keep_rows.unsqueeze(2).expand_as(x)
        # x_val = x * keep_mask.float()
        keep_nozero = out.sum(dim=(0, 2)) != 0
        # print("keep_nozero",keep_nozero)
        x_val = x[:,keep_nozero,:]
        # print("x_val",x_val[0,:,:])
        # print(x_val.shape)
        out = out[:,keep_nozero,:]
        out = self.out_proj(out)
        # print("out",out[0,:,:])
        # result =(out +x_val).sum(dim=1)/keep_rows.sum(dim=1).unsqueeze(-1)
        result = (out + x_val).mean(dim=1)
        return result
            

    
    def drop_and_reweight_compare_diagonal_threshold(self, attention, threshold=0.01):
        BHW, N, C = attention.shape
        attention_sum = attention.sum(dim=0)  # sum over the first dimension
        attention_sum_diagonal = attention_sum.diagonal(dim1=0, dim2=1)  # shape: (N,)
        # print("attention_sum_diagonal",attention_sum_diagonal)
        attn_diagonal = attention.diagonal(dim1=1, dim2=2)  # shape: (BHW, N)
        atten_diagonal_variance = attn_diagonal.var(dim=0)  # shape: (N,)
        # keep_diag = attention_sum_diagonal >= threshold * BHW
        keep_diag = atten_diagonal_variance <= threshold
        if not keep_diag.any():
            min_idx = atten_diagonal_variance.argmin()
            keep_diag = torch.zeros_like(keep_diag)
            keep_diag[min_idx] = True
        keep_mask = keep_diag.unsqueeze(1).expand(N, C)
        keep_mask = keep_mask& keep_mask.transpose(0, 1)
        keep_mask = keep_mask.unsqueeze(0).expand(BHW, N, C)  # shape: (BHW, N, C)
        # print('threshold',threshold)

        atten_masked = attention * keep_mask.float()
        atten_masked_for_sm = atten_masked.clone()
        # atten_masked_for_sm[atten_masked_for_sm == 0] = float('-inf')
        # attention = F.softmax(atten_masked_for_sm, dim=2)
        atten_masked_for_sm = atten_masked_for_sm / (atten_masked_for_sm.sum(dim=2, keepdim=True) + 1e-6)
        attention = torch.nan_to_num(atten_masked_for_sm, nan=0.0)
        return attention                     # row‐wise softmax  

class dropSelfattention(LightweightAttention):
    def forward(self, x):
        bhw, n, f = x.shape
        x = x + self.pos_emb  # add positional embedding
        
        # generate queries, keys, values
        qkv = self.proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'bhw n (h d) -> bhw h n d', h=self.num_heads), qkv)# h * d = f
        # calculate attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        """
        add when test robustness
        """
        ###############################################################################
        # drop and reweight
        attn, indice = self.drop_and_reweight(attn) # attn.shape = [BHW, h, q, k] indice.shape = [BHW, h, q, 1]
        # print("1st head reweight attn matrix in first pixel",attn[0,0,:,:])
        x_reshaped = rearrange(x, 'bhw n (h d) -> bhw n h d', h=self.num_heads)
        x_reshaped = x_reshaped.permute(0, 2, 1, 3) # [BHW, h, n, d]
        mask = torch.ones_like(x_reshaped, device=x_reshaped.device)
        mask.scatter_(dim=2, index=indice.expand(-1, -1, -1, self.dim_head), value=0)
        x_reshaped = x_reshaped * mask
        x_reshaped = x_reshaped.permute(0, 2, 1, 3) # [BHW, n, h, d]
        x_reshaped = rearrange(x_reshaped, 'bhw n h d -> bhw n (h d)')
        # aggregate values
    ########################################################################
        out = torch.matmul(attn, v)
        out = rearrange(out, 'bhw h n d -> bhw n (h d)')  
        result = (self.out_proj(out)+x).mean(dim=1) if n<=2 else (self.out_proj(out)+x_reshaped).sum(dim=1)/(n - 1)
        # result= (self.out_proj(out)+x).mean(dim=1)
        return result  # average over heads
    ###############################################################################
    def drop_and_reweight(self, attn_weights):
        bhw, h, q, k = attn_weights.shape
        
        min_indices = attn_weights.mean(dim=1).argmin(dim=-1)
        indices = min_indices.view(bhw, 1, q, 1).expand(-1, h, -1, -1)

        mask = torch.ones_like(attn_weights, device=attn_weights.device)
        mask.scatter_(dim=3, index=indices, value=0)  
        
        masked_attn = attn_weights * mask
        return masked_attn / (masked_attn.sum(dim=-1, keepdim=True) + 1e-6),indices
    


class EfficientDynamicRouter(nn.Module):
    def __init__(self, in_dim, out_dim,drop_rate=0.1,use_drop=False):
        super().__init__()
        if not use_drop:
            self.attn = LightweightAttention_singlehead(in_dim, drop_rate=drop_rate)
        else: 
            self.attn = LightweightAttention_singlehead_drop(in_dim, drop_rate=drop_rate)
            print("using dropSelfattention")
        self.conv_finale = nn.Conv1d(in_channels= in_dim , out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(in_dim)
        

    def forward(self, denoiser_outputs):
        """

        """
        features = torch.stack(denoiser_outputs, dim=1)  # [B, N, F, H, W]
        B, N, C, H, W = features.shape
        features = rearrange(features, 'b n c h w -> b h w n c')
        features = rearrange(features, 'b h w n c-> (b h w) n c')
        features = self.norm(features)
        # calculate  weights output
        weighted = self.attn(features)  # [BHW, C]
        # recover spatial information
        # spatial = rearrange(weighted, '(b h w) c -> b c h w', b=B, h=H, w=W)
        weighted_reshape = rearrange(weighted, " (B H W) C-> B H W C", B=B, H=H, W=W, C=C)
        weighted_reshape = rearrange(weighted_reshape, " B H W C-> (B H) C W ", B=B, H=H, W=W, C=C)
        out = self.conv_finale(weighted_reshape)
        out = rearrange(out, "(B H) C W -> B C H W", B=B, H=H)
        return out




