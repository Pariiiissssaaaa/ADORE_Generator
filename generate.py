import argparse
import torch
from torch.autograd import Variable
import data
from tqdm import tqdm
import json
from collections import defaultdict



aspect=['FOOD#QUALITY positive', 'RESTAURANT#PRICES positive', 'AMBIENCE#GENERAL negative', 'FOOD#QUALITY negative', 'RESTAURANT#PRICES negative', 'DRINKS#QUALITY positive', 'AMBIENCE#GENERAL positive', 'RESTAURANT#GENERAL negative', 'DRINKS#QUALITY negative', 'RESTAURANT#GENERAL positive']



#... Map each spect to a numerical value
def mapContextToIndex(context):
    mapDict=defaultdict()
    for i, item in enumerate(context):
        mapDict[item]=i
    return mapDict

mapAspect=mapContextToIndex(aspect)

save_1='model_rnn.pt'
save_2='model_context.pt'

inputt='./input'

# Model parameters.
checkpoint_rnn='./'+save_1
checkpoint_context='./'+ save_2
words=20
seed=1111
temperature=0.6
log_interval=100
n=10 #number of generated reviews for each aspect 

# Set the random seed manually for reproducibility.
#torch.manual_seed(seed)
cuda=True
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if cuda else "cpu")




corpus = data.Corpus(inputt)
ntokens = len(corpus.dictionary)

def generator(aSpect):
    
    idx=mapAspect[aSpect]
    context_id=torch.FloatTensor(aspect_one_hot[idx]).to(device)

 
    with open(checkpoint_rnn, 'rb') as f:
        model_1 = torch.load(f).to(device)
    f.close()

    with open(checkpoint_context, 'rb') as f:
        model_2 = torch.load(f).to(device)
    f.close()
    
    model_1.eval()
    model_2.eval()

    encoder_output=model_2(context_id)
    hidden = model_1.init_hidden(1)

    #...... Generator (Decoder) .......


    hidden=encoder_output.unsqueeze(0).expand_as(hidden)

    
    first_word_id=corpus.dictionary.word2idx['<sos>']
    input=torch.tensor([[first_word_id]]).to(device)
    word = corpus.dictionary.idx2word[1]

    #with open(outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        rev=''
        while (word!='<eos>'):

            output, hidden = model_1(input, hidden.contiguous(), encoder_output)

            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            
            if word!='<eos>':
                rev+=word+' '
    return rev

#...... Generate Review Samples .........
sample_reviews=defaultdict(list)
for label in tqdm(aspect):
    for i in range(n):
        sample_rev=generator(label)
        sample_reviews[label].append(sample_rev)
        
#..... Save as Json file .......
with open('generated_reviews.json', 'w') as f:
    json.dump(sample_reviews, f)