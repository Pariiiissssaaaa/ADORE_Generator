import torch
import torch.nn as nn
import torch.nn.functional as F


# ........... Attention method (Parisa)...........
def new_parameter(*size):
    out = nn.Parameter(torch.FloatTensor(*size))
    
    return out

#..... Create emb layer with pre-trianed embeddings (Parisa)......
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size(0), weights_matrix.size(1)
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer


#...... Attention Layer Class (Parisa).......
class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)
        self.attention.requires_grad=True
        torch.nn.init.xavier_normal_(self.attention)
        
        

    def forward(self, x_in):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score, dim=0).view(x_in.size(0), x_in.size(1), x_in.size(2), 1)
        scored_x = x_in * attention_score
        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=0)#instead of summation we can use a linear transformation

        return condensed_x
    
    
#....... Context Encoder Class (Parisa).........
class EncoderNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderNN, self).__init__()
        self.fc=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        
    def forward(self, x):
        out=self.fc(x)
        out=self.relu(out)
        
        return out
        

#....... Language Model Class (Modified).........
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, weights_matrix, dropout=0.5, tie_weights=True, attention_layer=True, emb_layer=True):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        
        #Parisa's Modification
        if  emb_layer:
            self.encoder=create_emb_layer(weights_matrix)
            
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
            
        #Parisa's Modification
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            
            
        self.decoder = nn.Linear(nhid, ntoken)
        
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight


        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        
        #Parisa's Modification
        if emb_layer==False:
            self.init_weights()
      
        self.attention_layer=attention_layer
        
        if self.attention_layer:
            self.attn=Attention(self.nhid)
        #Parisa's Modification
        

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, encoder_output):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        
        
        # Parisa's Modification to add Attection layer ......
        if self.attention_layer:
            encoder_output=encoder_output.unsqueeze(0).expand_as(output)
            output=output.view(1, output.size(0), output.size(1), output.size(2))
            encoder_output=encoder_output.view(1, encoder_output.size(0), encoder_output.size(1), encoder_output.size(2))
            attention_input=torch.cat((output, encoder_output), 0)
            output=self.attn(attention_input)
        # Parisa's Modification to add Attection layer ......
        
        
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
