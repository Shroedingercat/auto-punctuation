import numpy as np
import pickle
import string
import torch
import tqdm
import transformers as ppb
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
import torchtext



def to_tsv(path="Война_и_мир", split_size=64, lenta=True):
    """
    load, split and prepare text for training
    :param path:
    :param split_size:
    :return:
    """
    print("reading data...")
    with open(path, "r", encoding='utf-8') as fl:
        text = fl.read()
    print("tokenizing data...")
    text = text.replace('–\xa0', '')
    text = text.split('\n')
    cl_text = []
    for line in tqdm.tqdm(text, "tokenizing data...", len(text)):
        if len(line) != 0:
            cl_text.append(wordpunct_tokenize(line))        
    text = cl_text      
    
    #text = text[:int(0.2*len(text))]
    
    
    
    punct = {"<word>": 0, ".": 1, "!": 2, "?": 3, ":": 4, ",": 5, "-": 6, "<None>": 7}
    # Delete punctuation
    text_cl = []
    targets = []
    for i in tqdm.tqdm(range(len(text)), "split data...", len(text)):
        if len(text[i]) >= 5:
            targets.append([])
            text_cl.append([])
            curr_punct = False

            for j in range(len(text[i])):

                if curr_punct:
                    curr_punct = False
                    continue

                if j != len(text[i]) - 1:
                    if text[i][j+1] in punct:
                        targets[-1].append(text[i][j+1])
                        curr_punct = True
                    else:
                        targets[-1].append("<word>")
                else:
                    targets[-1].append("<word>")
                text_cl[-1].append(text[i][j])
    df = pd.DataFrame()
    text = [' '.join(line) for line in text_cl]
    df['text'] = text
    df["targets"] = [" ".join(line) for line in targets]
    df.to_csv("data.tsv", sep="\t", index=False, header=False)


def check_one(reference, hypothesis):
    correct = 0
    incorrect = 0
    ref = wordpunct_tokenize(reference)
    hyp = wordpunct_tokenize(hypothesis)
    ref_i, hyp_i = 0, 0
    punct_places = 0
    while ref_i < len(ref) and hyp_i < len(hyp):
        need_punct_check_ref = False
        need_punct_check_hyp = False
        cur_ref = ref[ref_i]
        if cur_ref in string.punctuation:
            need_punct_check_ref = True
            punct_places += 1
        cur_hyp = hyp[hyp_i]
        if cur_hyp in string.punctuation:
            need_punct_check_hyp = True
        if need_punct_check_ref and need_punct_check_hyp:
            if cur_ref == cur_hyp:
                correct += 1
            else:
                incorrect += 1
            ref_i += 1
            hyp_i += 1
            continue

        if need_punct_check_ref and not need_punct_check_hyp:
            incorrect += 1
            ref_i += 1
            continue

        if not need_punct_check_ref and need_punct_check_hyp:
            incorrect += 1
            hyp_i += 1
            continue

        assert cur_hyp == cur_ref, "The phrases are inconsistent!"
        ref_i += 1
        hyp_i += 1

    return correct/punct_places - incorrect/(2 * len(reference))


gold_str = "Начиная жизнеописание героя моего, Алексея Федоровича Карамазова, нахожусь в некотором недоумении. " \
           "А именно: хотя я и называю Алексея Федоровича моим героем, но, однако, сам знаю, что человек он " \
           "отнюдь не великий, а посему и предвижу неизбежные вопросы вроде таковых: чем же замечателен ваш " \
           "Алексей Федорович, что вы выбрали его своим героем?"
test_str = "Начиная жизнеописание героя моего, Алексея Федоровича Карамазова нахожусь в некотором недоумении. " \
           "А именно: хотя я и называю Алексея Федоровича моим героем, но однако, сам знаю, что человек он " \
           "отнюдь не великий, а посему, и предвижу неизбежные вопросы вроде таковых - чем же замечателен ваш " \
           "Алексей Федорович, что вы выбрали его своим героем?"









