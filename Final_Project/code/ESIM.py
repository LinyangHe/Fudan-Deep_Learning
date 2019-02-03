# python: 3.6
# encoding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
import numpy

class ESIM(nn.Module):
    def __init__(self, num_word, hidden_size, embed_dim, linear_size, dropout = 0.5, pretrained_weight = None):
        super().__init__()
        self.num_word = num_word
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.linear_size = linear_size
        self.dropout = dropout

        #word embedding layer, turn a word index into a wordvector
        self.embedding = nn.Embedding(self.num_word, self.embed_dim)
        #load pretrained word vectors such as GloVe
        if pretrained_weight is not None:
            # pretrained_weight should be a numpy.ndarray
            self.embedding.from_pretrained(torch.from_numpy(pretrained_weight), freeze=True)

        #Batchnormalize the embedding output.
        self.embedding_bn = nn.BatchNorm1d(self.embed_dim)
        # lstm1: Input encoding layer
        self.lstm1 = nn.LSTM(self.embed_dim, self.hidden_size,
                             batch_first=True, bidirectional=True)
        # lstm2: Inference composition layer 
        # 8: 4 ( [a, a', a-a', a.*a'] ) * 2( bidirectional )
        self.lstm2 = nn.LSTM(8 * self.hidden_size, self.hidden_size, 
                            batch_first=True, bidirectional=True)

        # the MLP classifier
        self.MLP = nn.Sequential(
            nn.BatchNorm1d(8 * self.hidden_size),
            nn.Linear(8 * self.hidden_size, self.linear_size),
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

    def soft_align(self, p_bar, h_bar, mask_p, mask_h):
        '''
        3p_bar: batch_size * p_seq_len * (2 * embed_dim)
        h_bar: batch_size * h_seq_len * (2 * embed_dim)
        mask_p: batch_size * p_seq_len
        mask_h: batch_size * h_seq_len 
        '''
        attention = torch.matmul(p_bar, h_bar.transpose(1, 2)) # batch_size * p_seq_len * h_seq_len

        # change '1.' in the mask tensor to '-inf'
        mask_p = mask_p.float().masked_fill_(mask_p, float('-inf'))
        mask_h = mask_h.float().masked_fill_(mask_h, float('-inf'))
        
        weight1 = F.softmax(attention + mask_h.unsqueeze(1), dim=-1) # batch_size * p_seq_len * h_seq_len
        weight2 = F.softmax(attention.transpose(1, 2) + mask_p.unsqueeze(1), dim=-1) # batch_size * h_seq_len * p_seq_len
        
        p_align = torch.matmul(weight1, h_bar)                       # batch_size * p_seq_len * (2 * embed_dim)
        h_align = torch.matmul(weight2, p_bar)                      # batch_size * h_seq_len * (2 * embed_dim)

        return p_align, h_align

    def pooling(self, v_p, v_h):
        # v_p: batch_size * p_seq_len * (2 * hidden_size)
        # v_h: batch_size * h_seql_len * (2 * hidden_size)

        p_avg = F.avg_pool1d(v_p.transpose(1, 2), v_p.shape[1]).squeeze(-1) # batch_size * (2 * hidden_size)
        p_max = F.max_pool1d(v_p.transpose(1, 2), v_p.shape[1]).squeeze(-1) # batch_size * (2 * hidden_size)

        h_avg = F.avg_pool1d(v_h.transpose(1, 2), v_h.shape[1]).squeeze(-1) # batch_size * (2 * hidden_size)
        h_max = F.max_pool1d(v_h.transpose(1, 2), v_h.shape[1]).squeeze(-1) # batch_size * (2 * hidden_size)

        v = torch.cat([p_avg, p_max, h_avg, h_max], -1) # batch_size * (8 * hidden_size)
        
        return v

    def forward(self, p_seq, h_seq):
        # p_seq: a word index sequence denoting premise, batch_size * p_seq_len 
        # h_seq: a word index sequcne denoting hypothesis, batch_size * h_seq_len
        '''
        Level0 - Embedding
        '''
        #p_embed: embedd premise word sequence, 
        p_embed = self.embedding(p_seq) # batch_size * p_seq_len * emb_dim 
        #h_embed: embedd hypothesis word sequence, 
        h_embed = self.embedding(h_seq) # batch_size * h_seq_len * emb_dim 

        '''
        Level0 - Batch Normalization
        '''
        # self.embed_bn() needs the input's shape as emb_dim
        # contiguous makes a deep copy.
        p_embed_bn = self.embedding_bn(p_embed.transpose(1, 2).contiguous()).transpose(1, 2) # batch_size * p_seq_len * emb_dim
        h_embed_bn = self.embedding_bn(h_embed.transpose(1, 2).contiguous()).transpose(1, 2) # batch_size * h_seq_len * emb_dim

        '''
        Level1 - Input encoding
        '''
        p_bar, _ = self.lstm1(p_embed_bn) # batch_size * p_seq_len * (2 * hidden_size)
        h_bar, _ = self.lstm1(h_embed_bn) # batch_size * h_seq_len * (2 * hidden_size)

        '''
        Level2 - Local Inference Modeling
        '''
        #We need the mask Tensor to help caculate the soft attention
        # mask_p: a mask Tensor recording if a word in p_seq is padding, 
        mask_p = p_seq.eq(0) # batch_size * p_seq_len
        # mask_h: a mask Tensor recording if a word in h_seq is padding, 
        mask_h = h_seq.eq(0) # batch_size * h_seq_len

        # Soft Align
        p_align, h_align = self.soft_align(p_bar, h_bar, mask_p, mask_h) # batch_size * seq_len * (2 * hidden_size)

        # Combine
        p_combined = torch.cat([p_bar, p_align, p_bar - p_align, p_bar * p_align], -1) # batch_size * p_seq_len * (8 * hidden_size)
        h_combined = torch.cat([h_bar, h_align, h_bar - h_align, h_bar * h_align], -1) # batch_size * h_seq_len * (8 * hidden_size)

        '''
        Level3 - Inference Composition
        '''
        v_p, _ = self.lstm2(p_combined) # batch_size * p_seq_len * (2 * hidden_size)
        v_h, _ = self.lstm2(h_combined) # batch_size * h_seq_len * (2 * hidden_size)

        # Pooling
        v = self.pooling(v_p, v_h)    # batch_size * (4 * hidden_size)

        # Classifier
        pred_label = self.MLP(v)      # batch_size * 3

        return {'pred': pred_label}
