# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import pandas as pd
import re
import gensim
import pickle
import os
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

def seed_everything(seed: int):  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def cleaner(text):
    clean_text = text.lower()
    clean_text = re.sub("\\n"," ",clean_text)
    clean_text = re.sub("\?",".",clean_text)
    clean_text = re.sub("!",".",clean_text)
    clean_text = re.sub("[^a-zA-Z.]", " ", clean_text)
    clean_text = re.sub("\."," </s> ",clean_text)
    if len(clean_text.split()) == 0:
        return ''

    return clean_text

def read_data(path):
    data = pd.read_csv(path)
    data['clean_text'] = data.text.apply(cleaner)
    data['clean_text'].replace('', np.nan, inplace=True)
    data.dropna( inplace=True)
    return data

def create_embedding(text, word2vec):
    vocab = set(text.split())
    print('Unique words in text:',len(vocab))
    custom_vocab = {}

    for word in vocab:
        try:
            custom_vocab[word] = word2vec.get_vector(word)
        except:
            pass

    custom_vocab['</s>'] = word2vec.get_vector('</s>')
    custom_vocab['<s>'] = np.random.randn(300)
    custom_vocab['<UNK>'] = np.mean(word2vec.vectors,axis=0)
    custom_vocab['<PAD>'] = np.zeros(300)

    embed_vectors = np.zeros((len(custom_vocab),300))
    word2id = {}
    id2word = {}

    for i,word in enumerate(custom_vocab.keys()):
        word2id[word] = i
        id2word[i] = word

        embed_vectors[i] = custom_vocab[word]

    print("Embedding size: ",np.shape(embed_vectors))
    return embed_vectors, word2id, id2word


def create_sentences(full_text, split_word=False):
    sents = full_text.split('</s>')
    clean_sents = []
    for sent in sents:
        if len(sent.split()) < 3:
            continue
        if split_word:
            s = []
            for word in sent.split():
                s.append(word.split(''))
        clean_sents.append(sent)
    return clean_sents

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, word2id, max_len=None, label=None):
        self.sentences = sentences
        self.word2id = word2id
        self.max_len = max_len
        self.label = label

    def __getitem__(self, idx):
        item = {}
        item['forward'] = [self.word2id['<s>']]
        item['reverse']  = []
        i = 0
        for w in self.sentences[idx].split():
            if self.max_len and i == self.max_len:
                break
            item['forward'].append(self.word2id[w] if w in self.word2id else self.word2id['<UNK>'])
            i += 1
        item['forward'].append(self.word2id['</s>'])
        item['reverse'] = torch.tensor(item['forward'][::-1])
        item['forward'] = torch.tensor(item['forward'])
        if self.label:
            item['label'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.sentences)


def pad_collate(batch):
  batch = sorted(batch, key=lambda item: item['forward'].size(0),reverse=True)
  x_fwd = [x['forward'] for x in batch]
  x_rev = [x['reverse'] for x in batch]
  x_lens = [len(x) for x in x_fwd]
  
  xfwd_pad = pad_sequence(x_fwd, padding_value=41900)
  xrev_pad = pad_sequence(x_rev, padding_value=41900)

  return xfwd_pad, xrev_pad, x_lens


class BiLSTM(nn.Module):

    def __init__(self, embed_layer):
        super(BiLSTM, self).__init__()

        self.vocab_size = embed_layer.weight.size(0)
        self.hidden_size = embed_layer.weight.size(1)
        self.embed_layer = embed_layer

        for param in self.embed_layer.parameters():
            param.requires_grad = False

        self.forward_lstm1 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.backward_lstm1 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.forward_lstm2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.backward_lstm2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size*2,self.vocab_size)

    def forward(self, input_fwd, input_rev, seq_lens):

        embed_seq_fwd = self.embed_layer(input_fwd) #seq_len, batch, embed_size
        embed_seq_rev = self.embed_layer(input_rev)

        seq_len = embed_seq_fwd.size(0)
        batch_size = embed_seq_fwd.size(1)

        outputs = torch.zeros((seq_len, batch_size, self.hidden_size*2), device=device)
        hidden_states = torch.zeros((2, seq_len, batch_size, self.hidden_size*2), device=device)
        
        fwd_packed = pack_padded_sequence(embed_seq_fwd, seq_lens, enforce_sorted=True)
        fwd_out1, (hf1, cf1) = self.forward_lstm1(fwd_packed)
        fwd_out1, output_lengths = pad_packed_sequence(fwd_out1)

        rev_packed = pack_padded_sequence(embed_seq_rev, seq_lens, enforce_sorted=True)
        rev_out1, (hr1, cr1) = self.forward_lstm1(rev_packed)
        rev_out1, output_lengths = pad_packed_sequence(rev_out1)

        for i in range(len(seq_lens)):
            hidden_states[0,:seq_lens[i],i:i+1,:] = torch.concat((fwd_out1[:seq_lens[i],i:i+1,:],rev_out1[:seq_lens[i],i:i+1,:]),dim=2)

        fwd_packed = pack_padded_sequence(fwd_out1, seq_lens, enforce_sorted=True)
        fwd_out2, (hf2, cf2) = self.forward_lstm2(fwd_packed)
        fwd_out2, output_lengths = pad_packed_sequence(fwd_out2)

        rev_packed = pack_padded_sequence(rev_out1, seq_lens, enforce_sorted=True)
        rev_out2, (hr2, cr2) = self.forward_lstm2(rev_packed)
        rev_out2, output_lengths = pad_packed_sequence(rev_out2)

        for i in range(len(seq_lens)):
            hidden_states[1,:seq_lens[i],i:i+1,:] = torch.concat((fwd_out2[:seq_lens[i],i:i+1,:],rev_out2[:seq_lens[i],i:i+1,:].flip(dims=[0])),dim=2)

        for i in range(len(seq_lens)):
            outputs[:seq_lens[i]-2,i:i+1,:] = torch.concat((fwd_out2[:seq_lens[i]-2,i:i+1,:],rev_out2[:seq_lens[i]-2,i:i+1,:].flip(dims=[0])),dim=2)

        outputs = self.linear(outputs)
        
        return outputs, hidden_states, embed_seq_fwd

def fit(model, train_loader, opt, loss_fn, epochs=1):

    model = model.to(device)
    model.train()

    for epoch in trange(epochs):
        loss_sum_per_epoch = 0.0
        num_batches = 0
        
        i =0
        for x_fwd, x_rev, x_lens in train_loader:
            # context = batch['context']
            # label = batch['label']
            #batch = batch.reshape(batch.size(1),batch.size(0)).to(device)
            #label = label.to(device)
            x_fwd = x_fwd.to(device)
            x_rev = x_rev.to(device)
            x_lens = torch.tensor(x_lens)

            opt.zero_grad()
            bilm_out, hidden, embed_seq = model(x_fwd, x_rev, x_lens)
            loss = loss_fn(bilm_out[:-1].view((bilm_out.size(0)-1)*bilm_out.size(1),bilm_out.size(2)),x_fwd[1:].view((x_fwd.size(0)-1)*x_fwd.size(1)))
            loss_sum_per_epoch += loss.detach()
            num_batches += 1
            loss.backward()
            opt.step()
            del x_fwd, x_rev, x_lens, bilm_out, hidden, embed_seq
            i += 1
            if i%10 == 0:
                print(i)
           
            
        loss_per_epoch = loss_sum_per_epoch.item()/num_batches
        print("Epoch #"+str(epoch+1)+" Loss: "+str(loss_per_epoch)+" Perplexity: "+str(np.exp(loss_per_epoch)))


class ELMo(nn.Module):

    def __init__(self, embed_layer, bilm_layer, w=None, gamma=None):
        super(ELMo, self).__init__()
        
        self.embed_size = embed_layer.weight.size(1)
        self.hidden_size = self.embed_size
        self.embed_layer = embed_layer

        self.biLM = bilm_layer

        for param in self.biLM.parameters():
            param.requires_grad = False
            
        if w:
            self.W = nn.Parameter(w,requires_grad=True).to(device)
        else:
            self.W = nn.Parameter(torch.tensor([1/3 for i in range(3)]), requires_grad=True).to(device)

        if gamma:
            self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)
        else:
            self.gamma = nn.Parameter(torch.ones(1), requires_grad=True).to(device)

    def set_w(self, w):
        self.W = nn.Parameter(w,requires_grad=False).to(device)

    def set_gamma(self, gamma):
        self.gamma = nn.Parameter(gamma,requires_grad=False).to(device)

    def forward(self, input_fwd, input_rev, seq_lens):
        bilm_out, hidden_states, embed_seq =  self.biLM(input_fwd, input_rev, seq_lens)
        ELMo_embedding = self.W[0] * torch.concat((embed_seq,embed_seq),dim=2)
        #print(ELMo_embedding.shape)
        #print(hidden_states.shape)
        
        #print(hidden_states.shape)
        for i in range(2):
            #print(hidden_states[i].shape)
            ELMo_embedding += self.W[i+1] * hidden_states[i]
        ELMo_embedding *= self.gamma

        return ELMo_embedding, bilm_out


def cpad_collate(batch):
  #print(batch)
  batch = sorted(batch, key=lambda item: item['forward'].size(0),reverse=True)
  x_fwd = [x['forward'] for x in batch]
  x_rev = [x['reverse'] for x in batch]
  labels = [x['label'] for x in batch]
  x_lens = [len(x) for x in x_fwd]

  xfwd_pad = pad_sequence(x_fwd, padding_value=41900)
  xrev_pad = pad_sequence(x_rev, padding_value=41900)

  return xfwd_pad, xrev_pad, labels, x_lens


class SentimentAnalyser(nn.Module):

    def __init__(self, embed_layer, bilm_layer, num_classes, elmo_w=None, elmo_gamma=None):
        super(SentimentAnalyser,self).__init__()
        
        self.embedding_size = embed_layer.weight.size(1)
        self.embedder = embed_layer
        self.num_classes = num_classes
        self.elmo = ELMo(self.embedder, bilm_layer, w=elmo_w, gamma=elmo_gamma)
        self.lstm = nn.LSTM(self.embedding_size*3,self.embedding_size)
        self.linear = nn.Linear(self.embedding_size,self.num_classes)

    def set_w(self, w):
        self.elmo.set_w(w)

    def set_gamma(self, gamma):
        self.elmo.set_gamma(gamma)

    def forward(self, input_fwd, input_rev, seq_lens):
        embed_input = self.embedder(input_fwd)
        #print(embed_input.shape)
        elmo_rep, _ = self.elmo(input_fwd, input_rev, seq_lens)
        #print("elmo",elmo_rep.shape)
        input = torch.concat((embed_input,elmo_rep),dim=2)
        packed_input = pack_padded_sequence(input, seq_lens, enforce_sorted=True)
        #print("concat",input.shape)
        out, (h_n, c_n) = self.lstm(packed_input)
        #print("h_n",h_n.shape)
        output = self.linear(h_n[0])

        return output

def fit_sentiment_analyser(model, train_loader, opt, loss_fn, epochs=1):
    
    model = model.to(device)
    model.train()

    for epoch in trange(epochs):
        loss_sum_per_epoch = 0.0
        num_batches = 0
        
        for xfwd_pad, xrev_pad, labels, x_lens in train_loader:

            xfwd_pad = xfwd_pad.to(device)
            xrev_pad = xrev_pad.to(device)
            labels = torch.tensor(labels).to(device)
            x_lens = torch.tensor(x_lens)

            opt.zero_grad()
            out = model(xfwd_pad, xrev_pad, x_lens)
            #print(out.shape)
            loss = loss_fn(out,labels)
            loss_sum_per_epoch += loss.detach()
            num_batches += 1
            loss.backward()
            opt.step()
            del out, xfwd_pad, xrev_pad, labels, x_lens
            
            
        loss_per_epoch = loss_sum_per_epoch.item()/num_batches
        print("Epoch #"+str(epoch+1)+" Loss: "+str(loss_per_epoch))


def evaluate_lm(model, loader, loss_fn = nn.CrossEntropyLoss(ignore_index=41900)):

    model.to(device)
    model.eval()

    loss_sum = 0.0
    num_batches = 0

    for x_fwd, x_rev, x_lens in loader:

        x_fwd = x_fwd.to(device)
        x_rev = x_rev.to(device)
        x_lens = torch.tensor(x_lens)

        bilm_out, hidden, embed_seq = model(x_fwd, x_rev, x_lens)
        loss = loss_fn(bilm_out[:-1].view((bilm_out.size(0)-1)*bilm_out.size(1),bilm_out.size(2)),x_fwd[1:].view((x_fwd.size(0)-1)*x_fwd.size(1)))
        loss_sum += loss.detach()
        num_batches += 1

        del x_fwd, x_rev, x_lens, bilm_out, hidden, embed_seq
        

    loss = loss_sum.item()/num_batches
    print(" Loss: "+str(loss)+" Perplexity: "+str(np.exp(loss)))

    
def eval_sentiment_analyser(model, test_loader):
    
    model = model.to(device)
    model.eval()

    
    Y_pred = []
    Y = []
    

    for xfwd_pad, xrev_pad, labels, x_lens in test_loader:

        xfwd_pad = xfwd_pad.to(device)
        xrev_pad = xrev_pad.to(device)
        #labels = torch.tensor(labels).to(device)
        x_lens = torch.tensor(x_lens)

        out = model(xfwd_pad, xrev_pad, x_lens)
        y_pred = list(torch.argmax(out,dim=-1).detach().cpu())
        Y_pred.extend(y_pred)
        Y.extend(labels)
        del out, xfwd_pad, xrev_pad, labels, x_lens
       
        
    #print(y,y_pred)
    return classification_report(Y, Y_pred),confusion_matrix(Y, Y_pred)

def main():

    DATA_PATH = '/content/anlp-assgn2-data/yelp-subset.dev.csv'
    ROOT_DIR = '/content/drive/MyDrive/A2'
     
    data = read_data(DATA_PATH)

    with open(ROOT_DIR+'/word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)

    embedder = nn.Embedding.from_pretrained(torch.load(ROOT_DIR+'/embedder_wt.pt',map_location=torch.device('cpu')))

    sentences = create_sentences(' '.join(data.clean_text))

    train_dataset = YelpDataset(sentences, word2id, 30)

    train_loader = DataLoader(train_dataset, batch_size=256,drop_last=True, collate_fn = pad_collate)

    bilm_model = BiLSTM(embedder)

    bilm_model.load_state_dict(torch.load(ROOT_DIR+"/Bilm.pth",map_location=torch.device('cpu')))
    
    evaluate_lm(bilm_model,train_loader)

    sa_model = SentimentAnalyser(embedder,bilm_model,5)

    sa_model.load_state_dict(torch.load(ROOT_DIR+"/sentiment_analyser_5.pth",map_location=torch.device('cpu')))

    ctest_dataset = YelpDataset(list(data.clean_text), word2id, max_len=150, label=list(data.label))

    ctest_loader = DataLoader(ctest_dataset, batch_size=256, drop_last=True, collate_fn=cpad_collate)

    test_cr,test_cm = eval_sentiment_analyser(sa_model, ctest_loader)

    print(test_cr)

if __name__ == '__main__':
    main()

    
