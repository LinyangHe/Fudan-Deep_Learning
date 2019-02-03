# python: 3.6
# encoding: utf-8
import torch
from torch import nn
import torch.nn.functional as F

class ESIM(nn.Module):
    def __init__(self, hidden_size, embeds_dim, linear_size, num_word):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeds_dim = embed_dim
        self.linear_size = linear_size
        self.num_word = num_word

        self.dropout = 0.5

        #word embedding layer, turn a index seq into a wordvector
        self.embeds = nn.Embedding(self.num_word, self.embed_dim)
        #Batchnormalize the embedding output.
        self.bn_embeds = nn.BatchNorm1d(self.embed_dim)
        # lstm1: Input encoding layer
        self.lstm1 = nn.LSTM(self.embed_dim, self.hidden_size,
                             batch_first=True, bidirectional=True)
        # lstm2 Inference composition layer 
        # 8: [a, a', a-a', a.*a'] 4 * 2( bidirectional )
        self.lstm2 = nn.LSTM(
            self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)

        # the MLP classifier
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, 3),
            nn.Softmax(dim=-1)
        )

    def load_pretrained_glove(self, pretrained_weight):
        self.embed.from_pretrained(
            torch.from_numpy(pretrained_weight), freeze=True)

    def soft_attention_align(self, p_bar, h_bar, mask_p, mask_h):
        '''
        p_bar: batch_size * p_seq_len * (2 * embed_dim)
        h_bar: batch_size * h_seq_len * (2 * embed_dim)
        mask_p: batch_size * p_seq_len
        mask_h: batch_size * h_seq_len 
        '''
        attention = torch.matmul(p_bar, h_bar.transpose(1, 2)) # batch_size * p_seq_len * h_seq_len

        # change '1.' in mask tensor to '-inf'
        mask_p = mask_p.float().masked_fill_(mask_p, float('-inf'))
        mask_h = mask_h.float().masked_fill_(mask_h, float('-inf'))
        
        weight1 = F.softmax(attention + mask_h.unsqueeze(1), dim=-1) # batch_size * p_seq_len * h_seq_len
        weight2 = F.softmax(attention.transpose(1, 2) + mask_p.unsqueeze(1), dim=-1) # batch_size * h_seq_len * p_seq_len

        p_align = torch.matmul(weight1, h_bar)                       # batch_size * p_seq_len * (2 * embed_dim)
        h_align = torch.matmul(weight2, p_bar)                      # batch_size * h_seq_len * (2 * embed_dim)

        return p_align, h_align

    def apply_multiple(self, x):
        # x: batch_size * seq_len * (2 * hidden_size)
        x_avg = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        x_max = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        x_cat = torch.cat([x_avg, x_max], -1) #batch_size * (4 * hidden_size)

        return x_cat

    def forward(self, p_seq, h_seq):
        # p_seq: a word index sequence denoting premise, batch_size * p_seq_len 
        # h_seq: a word index sequcne denoting hypothesis, batch_size * h_seq_len

        # FastNLP has alreay turned the numpy arrays or lists into Pytorch Tensor
        # and padded the tensor. The length would be the longest length in
        # current batch.

        '''
        Level0 - Embedding
        '''
        #p_embed: embedd premise word sequence, 
        p_embed = self.embeds(p_seq) # batch_size * p_seq_len * emb_dim 
        #h_embed: embedd hypothesis word sequence, 
        h_embed = self.embeds(h_seq) # batch_size * h_seq_len * emb_dim 

        '''
        Level0 - Batch Normalization
        '''
        # self.embed_bn() needs the input's shape as emb_dim
        # contiguous makes a deep copy.
        p_embed_bn = self.bn_embeds(p_embed.transpose(1, 2).contiguous()).transpose(1, 2) # batch_size * p_seq_len * emb_dim
        h_embed_bn = self.bn_embeds(h_embed.transpose(1, 2).contiguous()).transpose(1, 2) # batch_size * h_seq_len * emb_dim

        '''
        Level1 - Input encoding
        '''
        p_bar, _ = self.lstm1(p_embed_bn) # batch_size * p_seq_len * (2 * hidden_size)
        h_bar, _ = self.lstm1(h_embed_bn) # batch_size * h_seq_len * (2 * hidden_size)
        '''
        Level2 - Local Inference Modeling
        '''
        #We need the mask Tensor to help compute the soft attention
        # mask_p: a mask Tensor recording if a word in p_seq is padding, 
        mask_p = p_seq.eq(0) # batch_size * p_seq_len
        # mask_h: a mask Tensor recording if a word in h_seq is padding, 
        mask_h = h_seq.eq(0) # batch_size * h_seq_len

        # Soft Align
        p_align, h_align = self.soft_attention_align(p_bar, h_bar, mask_p, mask_h) # batch_size * seq_len * (2 * hidden_size)

        # Combine
        p_combined = torch.cat([p_bar, p_align, p_bar - p_align, p_bar * p_align], -1) # batch_size * p_seq_len * (8 * hidden_size)
        h_combined = torch.cat([h_bar, h_align, h_bar - h_align, h_bar * h_align], -1) # batch_size * h_seq_len * (8 * hidden_size)

        '''
        Level3 - Inference Composition
        '''
        p_compose, _ = self.lstm2(p_combined) # batch_size * p_seq_len * (2 * hidden_size)
        h_compose, _ = self.lstm2(h_combined) # batch_size * h_seq_len * (2 * hidden_size)

        # Pooling
        p_pooled = self.apply_multiple(p_compose)
        h_pooled = self.apply_multiple(h_compose)

        # Classifier
        v = torch.cat([p_pooled, h_pooled], -1)
        pred_label = self.fc(v)

        return {'pred': pred_label}
