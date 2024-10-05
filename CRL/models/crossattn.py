import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, image_embed, text_embed):
        # image_embed: (b, h, w, c)
        # text_embed: (b, 1, 1, c)
        b, h, w, c = image_embed.shape
        
        q = self.q_proj(image_embed).reshape(b, h*w, self.num_heads, self.head_dim).transpose(1, 2) 
        k = self.k_proj(text_embed).reshape(b, 1, self.num_heads, self.head_dim).transpose(1, 2) 
        v = self.v_proj(text_embed).reshape(b, 1, self.num_heads, self.head_dim).transpose(1, 2) 
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        attn_weights = torch.softmax(attn_weights, dim=-2) 
        
        attn_output = torch.matmul(attn_weights, v) 
        attn_output = attn_output.transpose(1, 2).reshape(b, h, w, c) 
        attn_output = self.out_proj(attn_output) 
        
        return attn_output

if __name__ ==  '__main__':
    b, h, w, c = 1, 64, 64, 256
    num_heads = 8

    image_embed = torch.randn(b, h, w, c)
    text_embed = torch.randn(b, 1, 1, c)

    cross_attn = CrossAttention(c, num_heads)
    output = cross_attn(image_embed, text_embed)

    print(output.shape) 