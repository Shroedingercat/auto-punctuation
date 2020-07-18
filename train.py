import pandas as pd
from utils import to_tsv, check_one
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchtext import data
from torchtext import datasets
from argparse import ArgumentParser

#import spacy
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize

import string
import time
import random
import torchtext
import sys

from model import BiLSTMWithChars



def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d",  default=None)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--train", "-tr", default=False, type=bool)
    
    return parser.parse_args()

def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm.tqdm(iterator, f"train", len(iterator), False):
        
        text = batch.word
        chars = batch.char
        tags = batch.trg
        
        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text, chars)
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions, tags)
                
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.word
            chars = batch.char
            tags = batch.trg
            
            predictions = model(text, chars)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


    
def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def ai(gold_str, CHAR, WORD, TRG, model, device):
    char_level = CHAR.preprocess(gold_str)
    max_len = 0
    for i in range(len(char_level)):
        char_level[i] = ["<bos>"] + char_level[i] + ["<eos>"]
        if len(char_level[i]) > max_len:
            max_len = len(char_level[i])

    for i in range(len(char_level)):
        char_level[i] += ["<pad>"]*(max_len-len(char_level[i]))

    char_level = np.array(char_level)
    char_level = char_level.reshape(1, char_level.shape[0], char_level.shape[1])
    char_level = CHAR.numericalize(np.array(char_level))
    word_level = WORD.numericalize(np.array(WORD.tokenize(gold_str)).reshape(1,-1))
    word_level = word_level.reshape(-1, 1)
    model.eval()

    predictions = model(word_level.to(device), char_level.to(device))
    predictions = predictions.argmax(2).cpu().data.tolist()
    re_punct = {TRG.vocab.stoi[key]:key for key in TRG.vocab.stoi}

    tok = WordPunctTokenizer()
    txt = tok.tokenize(gold_str)
    punct_text = []
    for i in range(len(txt)):
        punct_text.append(txt[i])
        if predictions[i][0] != TRG.vocab.stoi["<word>"]:
            punct_text.append(re_punct[predictions[i][0]])
    return punct_text

    

def main(args):
    to_tsv(args.data, 64, False)
    df = pd.read_csv("data.tsv", sep="\t", header=None)
    
    punct = {"<word>": 0, ".": 0, "!": 0, "?": 0, ":": 0, ",": 0, "-": 0, "<None>": 0}
    for i in range(df.shape[0]):
        for word in punct:
            punct[word] += df[1][i].count(word)
    
    # Now lets try both word and character embeddings
    # Now lets try both word and character embeddings
    
    tokenizer = WordPunctTokenizer()
    def tokenize(x, tokenizer=tokenizer):
        return tokenizer.tokenize(x)

    WORD = data.Field(lower = False, tokenize=tokenize)
    TRG = data.Field(unk_token = None)

    # We'll use NestedField to tokenize each word into list of chars
    CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>")
    CHAR = data.NestedField(CHAR_NESTING)#, init_token="<bos>", eos_token="<eos>")

    fields = [(('word', 'char'), (WORD, CHAR)), ('trg', TRG)]
    dataset = data.TabularDataset(
        path="data.tsv",
        format='tsv',
        fields=fields
    )
    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
    MIN_FREQ = 3
    vec_ru = torchtext.vocab.Vectors("ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize.vec")
    WORD.build_vocab(
        train_data,
        min_freq = MIN_FREQ,
        vectors=vec_ru,
        unk_init = torch.Tensor.normal_
    )

    CHAR.build_vocab(train_data)
    TRG.build_vocab(train_data)
    
    BATCH_SIZE = args.batch_size

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _len_sort_key(x):
        return len(x.word)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, 
        device = device,
        sort_key=_len_sort_key
    )
    
    INPUT_DIM = len(WORD.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 200
    CHAR_INPUT_DIM = len(CHAR.vocab)
    CHAR_EMBEDDING_DIM = 30
    CHAR_HIDDEN_DIM = 30
    OUTPUT_DIM = 7
    N_LAYERS = 4
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    PAD_IDX = WORD.vocab.stoi[WORD.pad_token]
    
    model = BiLSTMWithChars(
        INPUT_DIM, 
        EMBEDDING_DIM,
        CHAR_INPUT_DIM,
        CHAR_EMBEDDING_DIM,
        CHAR_HIDDEN_DIM,
        HIDDEN_DIM, 
        OUTPUT_DIM, 
        N_LAYERS, 
        BIDIRECTIONAL, 
        DROPOUT, 
        PAD_IDX
    )
    
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean = 0, std = 0.1)

    model.apply(init_weights)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    weight = []
    for key in TRG.vocab.stoi:
        if key in punct:
            weight.append(1-(punct[key])/sum(punct.values()))
    
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    TAG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)
    
    N_EPOCHS = args.epochs

    best_valid_loss = float('inf')
    cleansed = filter(lambda x: x not in string.punctuation, word_tokenize(gold_str))
    hypothesis = ai(" ".join(cleansed), CHAR, WORD, TRG, model, device)
    
    if args.train:

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)

            #end_time = time.time()

            #epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best-model.pt')

            
            #print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        torch.save(model.state_dict(), 'best-model.pt')
            
    model.load_state_dict(torch.load("best-model.pth", map_location=device), )
    cleansed = filter(lambda x: x not in string.punctuation, word_tokenize(gold_str))
    hypothesis = ai(" ".join(cleansed), CHAR, WORD, TRG, model, device)
    hypothesis = " ".join(hypothesis)
    print(hypothesis)
    print("Quality: {:.2f}%".format(check_one(gold_str, hypothesis) * 100))
    
gold_str = "Начиная жизнеописание героя моего, Алексея Федоровича Карамазова, нахожусь в некотором недоумении. " \
           "А именно: хотя я и называю Алексея Федоровича моим героем, но, однако, сам знаю, что человек он " \
           "отнюдь не великий, а посему и предвижу неизбежные вопросы вроде таковых: чем же замечателен ваш " \
           "Алексей Федорович, что вы выбрали его своим героем?"    
    
if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))


