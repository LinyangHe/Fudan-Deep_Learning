import torch
import torch.nn as nn
import torch.nn.functional as F

def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

class PointerNetwork(nn.Module):
    def __init__(self, input_size, weight_size, hidden_size, is_GRU=False):
        super().__init__()
        self.input_size = input_size
        self.weight_size = weight_size
        self.hidden_size = hidden_size
        self.is_GRU = is_GRU

        if self.is_GRU:
            RNN = nn.GRU
            RNNCell = nn.GRUCell
        else:
            RNN = nn.LSTM
            RNNCell = nn.LSTMCell

        self.encoder = RNN(input_size, hidden_size, batch_first=True)
        self.decoder = RNNCell(input_size, hidden_size)
        
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.vt = nn.Linear(weight_size, 1, bias=False)

    def forward(self, input):
        batch_size = input.shape[0]
        decoder_seq_len = input.shape[1]

        # Encoding
        encoder_output, hc = self.encoder(input) 

        # Decoding states initialization
        hidden = encoder_output[:, -1, :] #hidden state for decoder is last timestep's output of encoder 
        if not self.is_GRU: #For LSTM, cell state is the sencond state output
            cell = hc[1][-1, :, :]
        decoder_input = to_cuda(torch.rand(batch_size, self.input_size))  
        
        # Decoding with attention             
        probs = []
        encoder_output = encoder_output.transpose(1, 0) #Transpose the matrix for mm
        for i in range(decoder_seq_len):  
            if self.is_GRU:
                hidden = self.decoder(decoder_input, hidden) 
            else:
                hidden, decoder_hc = self.decoder(decoder_input, (hidden, cell)) 
            # Compute attention
            sum = torch.tanh(self.W1(encoder_output) + self.W2(hidden))    
            out = self.vt(sum).squeeze()        
            out = F.log_softmax(out.transpose(0, 1).contiguous(), -1)  
            probs.append(out)

        probs = torch.stack(probs, dim=1)           
        return probs