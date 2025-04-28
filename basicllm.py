import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) *math.sqrt(self.d_model)
    


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model:int , seq_length:int ):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
                

        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        
        pe = pe.unsqueeze(0) # [1, L, D]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(True)
        return x # [batch, seq_len, d_model]
    
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps:float = 10*-6 ):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
class FeedForward(nn.Module):
    
    def __init__(self, d_model:int, d_ff:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(self.linear_1(x)))
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model:int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.h = h
        
        assert d_model%h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model//h
        
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model) 
        self.w_q = nn.Linear(d_model, d_model) 
        
        self.w_o = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        
        d_k = query.shape[-1]
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        output = attention_scores @ value
        return output, attention_scores
        

    def forward(self, q, k, v, mask):
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)
        
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(d_model, vocab_size)
        
        
    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos_embed: PositionalEncoding, tgt_pos_embed: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed 
        self.tgt_embed = tgt_embed
        self.src_pos_embed = src_pos_embed
        self.tgt_pos_embed = tgt_pos_embed
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_embed(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformers(src_vocab_size:int, tgt_vocab_size:int, src_seq_length:int, tgt_seq_length:int, d_model:int = 512, N:int = 24, h:int = 8, d_ff:int = 2048, dropout:float = 0.1):
    
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    src_pos_embed = PositionalEncoding(d_model, src_seq_length)
    tgt_pos_embed = PositionalEncoding(d_model, tgt_seq_length)
    
    encoder_blocks = []
    for _ in range(N):
        attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder = EncoderBlock(attention_block, feed_forward, dropout)
        encoder_blocks.append(encoder)
        
    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder = DecoderBlock(self_attention, cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder)
        
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_embed, tgt_pos_embed, projection_layer)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1: # Things like alpha, bias etc..
            nn.init.xavier_uniform_(p)
            
    return transformer