import os
import torch
import nltk
from nltk.tokenize import word_tokenize

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.target_vocab=set()
        #Parisa's Modification  add '<PAD>' as a token
        self.idx2word.append('<PAD>')
        self.word2idx['<PAD>']=0
        #Parisa's Modification

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            #Parisa's Modification
            self.target_vocab.add(word)
            #Parisa's Modification
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.maxSentenceLen=0
        self.longestSentence=[]
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        
        

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            
            for i, line in enumerate(f):
                
                #Parisa's Modification 
                words = ['<sos>'] + line.split() + ['<eos>']
            
                if len(words)>self.maxSentenceLen:
                
                    self.maxSentenceLen=len(words)
                    self.longestSentence=words
                #Parisa's Modification
                
                    
                for word in words:
                    self.dictionary.add_word(word)
       
        #Parisa's Modification
        tokens=(i+1)*self.maxSentenceLen
       #Parisa's Modification

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                
                #Parisa's Modification
                words = ['<sos>']+ line.split() + ['<eos>']
                if len(words)<self.maxSentenceLen:
                    words+=['<PAD>']*(self.maxSentenceLen - len(words))
                #Parisa's Modification
                
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
