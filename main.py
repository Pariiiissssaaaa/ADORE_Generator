import argparse
import time
import math
import os
import torch
import torch.nn as nn
from torch import optim
import torch.onnx
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

import data
import model


import nltk
from nltk.tokenize import word_tokenize
import re
from math import gcd

###############################################################################
# The language model code is adapted from "PyTorch word level language modeling example": https://github.com/pytorch/examples/tree/master/word_language_model".

# The modifications throughout main.py, data.py and model.py are marked as "parisa's Modification"
# The context encoder, regularization, attention layer and embedding layer are originally develped for this project by Parisa. 
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Aspect/Sentiment specific RNN Yelp review generator')

parser.add_argument('--inputt', type=str, default='./input',
                    help='location of the data corpus')
parser.add_argument('--model_type', type=str, default='GRU',
                    help='The context encoder is designed for GRU architecture (GRU)')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save_1', type=str, default='model_rnn.pt',
                    help='path to save the generator model')
parser.add_argument('--save_2', type=str, default='model_context.pt',
                    help='path to save the context encoder model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
#... The following arguments ease ablation study.....
parser.add_argument('--attention', action='store_true',
                    help='use attention to predict next word')
parser.add_argument('--regularizer', action='store_true',
                    help='use regularization term to address rare words')
parser.add_argument('--preTrainedEmb', action='store_true',
                    help='use pre-trained word embedding to initialize emb layer')


args = parser.parse_args()

epochs=args.epochs
learning_rate=args.lr

save_1=args.save_1
save_2=args.save_2

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

print (args)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.inputt)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bptt, bsz):
    #parisa's Modification
    lcm=int(bptt*bsz)
    print ('number of tokens in data tensor for each batch is {}'.format(lcm))
    #Parisa's Modification
    
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // lcm
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * lcm)
    
    #Parisa's Modification
    # Evenly divide the data across the bsz batches.
    data = data.view(-1, bptt).contiguous()
    #Parisa's Modification
    
    return data.to(device)


batch_size=args.batch_size
eval_batch_size = batch_size
bptt=corpus.maxSentenceLen
train_data = batchify(corpus.train, bptt, batch_size)
val_data = batchify(corpus.valid, bptt, batch_size)
test_data = batchify(corpus.test, bptt, batch_size)


###############################################################################
# Data preparation for context encoder network (Parisa)
###############################################################################


aspect=['FOOD#QUALITY positive', 'RESTAURANT#PRICES positive', 'AMBIENCE#GENERAL negative', 'FOOD#QUALITY negative', 'RESTAURANT#PRICES negative', 'DRINKS#QUALITY positive', 'AMBIENCE#GENERAL positive', 'RESTAURANT#GENERAL negative', 'DRINKS#QUALITY negative', 'RESTAURANT#GENERAL positive']



#... Map each spect to a numerical value ... 
def mapContextToIndex(context):
    mapDict=defaultdict()
    for i, item in enumerate(context):
        mapDict[item]=i
    return mapDict

mapAspect=mapContextToIndex(aspect)



#.. convert each aspect to a one-hot vector 
def indices_to_one_hot(context, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    data=[]
    for i in range(nb_classes):
        data.append(i)
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


aspect_one_hot=indices_to_one_hot(aspect, len(aspect))




def Build_contextTensor(inputFile, num, mapAspect):
    
    context_vector=[]
    
    f=open(inputFile)
    for i, line in enumerate(f):
        line=line.strip()
        idx_aspect=mapAspect[line]
        context_vector.append(list(aspect_one_hot[idx_aspect]))
    
    return torch.FloatTensor(context_vector).to(device)
        

train_context=Build_contextTensor(args.inputt+'/train_label.txt', len(train_data), mapAspect)
val_context=Build_contextTensor(args.inputt+'/valid_label.txt', len(val_data), mapAspect)
test_context=Build_contextTensor(args.inputt+'/test_label.txt', len(test_data), mapAspect)

###############################################################################
# Data preparation for context encoder network (Parisa)
###############################################################################
   
    
###############################################################################
# Building Embedding Tensor to initialize Emb layer (Parisa)
###############################################################################

import bcolz
import numpy as np

words_preTrained=[]
idx=0
word2idx_preTrained={}
vectors=bcolz.carray(np.zeros(1), mode='w')

with open ('restaurant_emb.vec', 'rb') as f:
    for i, l in enumerate(f):
        if i>0:
            line=l.decode().split()
            word=line[0]
            words_preTrained.append(word)
            word2idx_preTrained[word]=idx
            idx+=1
            vect=np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            

#Reshape bcolz vectors
vectors = bcolz.carray(vectors[1:].reshape((len(words_preTrained), len(vect))), mode='w')
vectors.flush()

# map words to vectors
glove = {w: vectors[word2idx_preTrained[w]] for w in words_preTrained}


matrix_len=len(corpus.dictionary.word2idx)
emb_dim=len(vect)
weights_matrix=np.zeros((matrix_len, emb_dim))
words_found=0

for i, word in enumerate(corpus.dictionary.word2idx):
    try:
        weights_matrix[i]=glove[word]
        words_found+=1
    except KeyError:
        weights_matrix[i]=np.random.normal(scale=0.6, size=(emb_dim, ))
        
        
print ('vocab size is {}'.format(len(corpus.dictionary.word2idx)), '\nnumber of words w/ emb vector is {}'.format(words_found) , '\nnumber of words w/o emb vector is {}'.format(len(corpus.dictionary.word2idx)-words_found))

#convert numpy matrix to tensor
weights_matrix = torch.from_numpy(weights_matrix)

###############################################################################
# Building Embedding Tensor to initialize Emb layer (Parisa)
###############################################################################


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model_1 = model.RNNModel(args.model_type, ntokens, args.emsize, args.nhid, args.nlayers, weights_matrix, args.dropout, args.tied, args.attention, args.preTrainedEmb).to(device)
model_2= model.EncoderNN(len(aspect), args.nhid).to(device)

criterion = nn.CrossEntropyLoss()

model_1_optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)
model_2_optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)

print ('Review Generator Architecture is \n {}'.format(model_1))
print ('Context Encoder Architecture is \n {}'.format(model_2))


###############################################################################
# Calculating Regularization Term (Parisa)
###############################################################################
def calculate_Regularization_Term(m, N, NN):
    

    m_norm=F.normalize(m, p=2, dim=1)# normalize input tensor on dimension 1
    m_dot_product = torch.mm(m_norm, m_norm.t())# Matrix multiplication
    #Regularization_value=(np.matrix(m_dot_product.detach()).sum()-N)/NN # this operation is tested
    Regularization_value=(torch.sum(m_dot_product).item()-N)/NN # this operation is tested

    return Regularization_value

###############################################################################
# Calculating Regularization Term (Parisa)
###############################################################################


###############################################################################
# Training code
###############################################################################

def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchfutrnction, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source, context_source):
    # Turn on evaluation mode which disables dropout.
    model_1.eval()
    model_2.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model_1.init_hidden(eval_batch_size)
   
    with torch.no_grad():
        
       
        for batch, x_batch in enumerate(chunks(data_source, eval_batch_size)):
            data=x_batch.t()
            l=data.size(1)
            targets=torch.cat((data[1:], data[-1].view(1,l)), 0).view(-1)
            data_context=context_source[batch*eval_batch_size:batch*eval_batch_size+eval_batch_size]
            
            encoder_output=model_2(data_context)
            hidden=encoder_output.unsqueeze(0).expand_as(hidden)
            
            output, hidden = model_1(data, hidden.contiguous(), encoder_output)
            
            
            total_loss+= len(data)*criterion(output.view(-1, ntokens), targets).item()
            
            
    return total_loss / (len(data_source)-1)


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(onnx_export)))
    model_1.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model_1.init_hidden(batch_size)
    torch.onnx.export(model_1, (dummy_input, hidden), path)


# ..... Added for this project ..........    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
#batching
def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
           
        
    
    
def train():
    # Turn on training mode which enables dropout.
    model_1.train()
    model_2.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    tmp_hidden=model_1.init_hidden(batch_size)
    

    #...... Parisa's Modification ....... 
    for batch, x_batch in enumerate(chunks(train_data, batch_size)): #batchify the context as well
        #...... Build data tensors ........
        data=x_batch.t()
        l=data.size(1)
        
        targets=torch.cat((data[1:], data[-1].view(1,l)), 0).view(-1)
        
        data_context=train_context[batch*batch_size:batch*batch_size+batch_size]
    
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        
        
            
        #......... Run Models ............
        model_1.zero_grad()
        model_2.zero_grad()
        
        encoder_output=model_2(data_context)
        hidden=encoder_output.unsqueeze(0).expand_as(tmp_hidden)
        output, hidden = model_1(data, hidden.contiguous(), encoder_output)
    
    
        #......... Calculate Loss ..........
        loss_1 = criterion(output.view(-1, ntokens), targets)
        
        if args.regularizer:
            m=model_1.encoder.weight
            N=m.shape[0]
            NN=np.square(N)

            loss_2=calculate_Regularization_Term(m, N, NN)
        else:
            loss_2=0.0
            
        loss=loss_1+loss_2
        #...... Parisa's Modification ....... 
        
        #........ update parameters ..........
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model_1.parameters(), args.clip)

        model_1_optimizer.step()
        model_2_optimizer.step()

        
        total_loss += loss.item()
        
         
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time

            total_loss = 0
            start_time = time.time()
    
    
# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
      
        train()
        
        val_loss = evaluate(val_data, val_context)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save_1, 'wb') as f:
                torch.save(model_1, f)
            with open(save_2, 'wb') as f:     
                torch.save(model_2, f)
            best_val_loss = val_loss


        #.... Adjust Learning Rate every 5 epoch ........    
        adjust_learning_rate(model_1_optimizer, epoch)
        adjust_learning_rate(model_2_optimizer, epoch)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    
    
# Load the best saved model.
with open(save_1, 'rb') as f:
    model_1 = torch.load(f)
f.close()

with open(save_2, 'rb') as f:
    model_2 = torch.load(f)
f.close()

# after load the rnn params are not a continuous chunk of memory
# this makes them a continuous chunk, and will speed up forward pass
model_1.rnn.flatten_parameters()


# Run on test data.
test_loss = evaluate(test_data, test_context)

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
