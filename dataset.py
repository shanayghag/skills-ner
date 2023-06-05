import pandas as pd
import pickle
import torch

from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["tokens"].values.tolist(), s["tags"].values.tolist())]
        self.grouped = self.data.groupby("sentences").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

if __name__ == '__main__':
    df = pd.read_csv("data/train.csv")
    df = df.dropna()
    # Get full document data struce
    getter = SentenceGetter(df)

    # Get sentence data
    sentences = [[tup[0] for tup in word_tag_tups] for word_tag_tups in getter.sentences]
    with open("data/Multi-sentence Cached data/sentences.pb", 'wb') as f:
        pickle.dump(sentences, f)
    f.close()

    # Get labels
    labels = [[s[1] for s in sent] for sent in getter.sentences]
    with open("data/Multi-sentence Cached data/labels.pb", 'wb') as f:
        pickle.dump(labels, f)
    f.close()

    f = open("data/Multi-sentence Cached data/sentences.pb", 'rb')
    sentences = pickle.load(f)
    f.close()

    f = open("data/Multi-sentence Cached data/labels.pb", 'rb')
    labels = pickle.load(f)
    f.close()

    print(sentences[0])
    print(labels[0])

    tags_vals = set(df["tags"].values)
    # Add X  label for word piece support
    # Add [CLS] and [SEP] as BERT need
    tags_vals.add('X')
    tags_vals.add('[CLS]')
    tags_vals.add('[SEP]')
    tags_vals.add('[PAD]')

    with open("data/Multi-sentence Cached data/tag_vals.pb", 'wb') as f:
        pickle.dump(tags_vals, f)
    f.close()

    tag2idx = {t: i for i, t in enumerate(tags_vals)}

    with open("data/Multi-sentence Cached data/tag2idx.pb", 'wb') as f:
        pickle.dump(tag2idx, f)
    f.close()

    idx2tag = {tag2idx[key]: key for key in tag2idx.keys()}

    with open("data/Multi-sentence Cached data/idx2tag.pb", 'wb') as f:
        pickle.dump(idx2tag, f)
    f.close()

    f = open("data/Multi-sentence Cached data/tag_vals.pb", 'rb')
    tag_vals = pickle.load(f)
    f.close()

    f = open("data/Multi-sentence Cached data/tag2idx.pb", 'rb')
    tag2idx = pickle.load(f)
    f.close()

    f = open("data/Multi-sentence Cached data/idx2tag.pb", 'rb')
    idx2tag = pickle.load(f)
    f.close()

    # Sanity checks
    print(tag2idx)
    print(idx2tag)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    tokenized_texts = []
    word_piece_labels = []

    i = 0
    for word_list, label_list in zip(sentences, labels):
        temp_tokens = []
        temp_labels = []

        # add [CLS] token at the front
        temp_tokens.append('[CLS]')
        temp_labels.append('[CLS]')

        # specialized data structure:
        # instead of labelling all word piece tokens with the same label
        # the first word piece token is given the original label
        # and the rest are labelled as 'X'
        for word, label in zip(word_list, label_list):
            tokenized_word = tokenizer.tokenize(word)
            for idx, token in enumerate(tokenized_word):
                temp_tokens.append(token)
                if idx == 0:
                    temp_labels.append(label)
                else:
                    temp_labels.append('X')

        # add [SEP] token at the end
        temp_tokens.append('[SEP]')
        temp_labels.append('[SEP]')

        tokenized_texts.append(temp_tokens)
        word_piece_labels.append(temp_labels)

        if i < 3:
            print(i, '\nTokens:', temp_tokens, '\n')
            print('Labels:', temp_labels)
            print('_' * 100, '\n')

        i += 1

    max_len = 40

    # convert tokens to ids
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype='long', truncating='post', padding='post')

    print(input_ids[0])

    # convert labels to ids
    tags = pad_sequences([[tag2idx[l] for l in label] for label in word_piece_labels],
                         maxlen=max_len, dtype='long', padding='post', truncating='post',
                         value=tag2idx['[PAD]'])

    print(tags[0])

    with open("data/Multi-sentence Cached data/input_ids.pb", 'wb') as f:
        pickle.dump(input_ids, f)
    f.close()

    with open("data/Multi-sentence Cached data/tags.pb", 'wb') as f:
        pickle.dump(tags, f)
    f.close()

    with open("data/Multi-sentence Cached data/tokenized_texts.pb", 'wb') as f:
        pickle.dump(tokenized_texts, f)
    f.close()

    with open("data/Multi-sentence Cached data/word_piece_labels.pb", 'wb') as f:
        pickle.dump(word_piece_labels, f)
    f.close()

    # Mask to avoid performing attention on padding token indices.
    # 1 for actual tokens, 0 for padded tokens.
    attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]

    with open("data/Multi-sentence Cached data/attention_masks.pb", 'wb') as f:
        pickle.dump(attention_masks, f)
    f.close()

    train_inputs, test_inputs, train_tags, test_tags, train_attn_masks, test_attn_masks = train_test_split(input_ids,
                                                                                                           tags,
                                                                                                           attention_masks,
                                                                                                           random_state=42,
                                                                                                           shuffle=True,
                                                                                                           test_size=0.2)

    print('train:', len(train_inputs), len(train_tags), len(train_attn_masks))
    print('test:', len(test_inputs), len(test_tags), len(test_attn_masks))

    with open("data/Multi-sentence Cached data/train_inputs.pb", 'wb') as f:
        pickle.dump(train_inputs, f)
    f.close()

    with open("data/Multi-sentence Cached data/test_inputs.pb", 'wb') as f:
        pickle.dump(test_inputs, f)
    f.close()

    with open("data/Multi-sentence Cached data/train_tags.pb", 'wb') as f:
        pickle.dump(train_tags, f)
    f.close()

    with open("data/Multi-sentence Cached data/test_tags.pb", 'wb') as f:
        pickle.dump(test_tags, f)
    f.close()

    with open("data/Multi-sentence Cached data/train_attn_masks.pb", 'wb') as f:
        pickle.dump(train_attn_masks, f)
    f.close()

    with open("data/Multi-sentence Cached data/test_attn_masks.pb", 'wb') as f:
        pickle.dump(test_attn_masks, f)
    f.close()





