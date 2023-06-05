import pandas as pd
import re
import nltk
import csv
import os
import fnmatch
import random
random.seed(42)

def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

# Load data and remove rows having columns less than 3
with open('data/job_test.csv', 'r') as f:
    csv_read = csv.reader(f)
    job_test = []
    for row in csv_read:
        if len(row) == 3:
            job_test.append(row)

with open('data/job_test.csv', 'w') as f:
    csv_write = csv.writer(f)
    for row in job_test:
        csv_write.writerow(row)


# cleaning the 'job' dataframe
job = pd.read_csv('data/job_train.csv')
job = job.append(pd.read_csv('data/job_test.csv'))
cv = pd.read_csv('data/cv_test.csv')

print ("Length of job dataframe: {}".format(len(job)))

sentences = []
tokens = []
tags = []

sent_cnt = 0

for sent, soft_skill in zip(job['context'], job['soft_skill']):
    xxx_cnt = find_all_indexes(sent, 'xxx')
    xxx_cnt = len(xxx_cnt)

    sent_cnt += 1
    i = 0
    if len(soft_skill.split()) == xxx_cnt:
        for word in nltk.word_tokenize(sent):
            sentences.append('Sentence: ' + str(sent_cnt))
            if word == 'xxx':
                tokens.append(soft_skill.split()[i])
                if i == 0:
                    tags.append('B-softskill')
                else:
                    tags.append('I-softskill')
                i += 1
            else:
                tokens.append(word)
                tags.append('O')

print(len(sentences), len(tokens), len(tags))

# cleaning the 'cv' dataframe
for sent in cv['sentence']:
    start_pos = sent.find('<b>')
    end_pos = sent.find('</b>') - len('</b>') + 1

    sent_cleaned = sent.replace('<b>', '').replace('</b>', '')
    soft_skill_tokens = sent_cleaned[start_pos: end_pos].split()
    sent_tokens = nltk.word_tokenize(sent_cleaned)

    sent_cnt += 1

    n_token = 0
    for token in sent_tokens:
        sentences.append('Sentence: ' + str(sent_cnt))
        tokens.append(token)
        if token in soft_skill_tokens:
            n_token += 1
            if n_token == 1:
                tags.append('B-softskill')
            else:
                tags.append('I-softskill')
        else:
            tags.append('O')

print(len(sentences), len(tokens), len(tags))

# Working on hobby text files!
for file_name in os.listdir('data/Data for NER/'):

    if fnmatch.fnmatch(file_name, '*.txt'):
        print("Working on :", file_name)

        f = open('data/Data for NER/' + file_name, 'r')
        lines = f.readlines()
        lines = random.sample(lines, len(lines))

        for line in lines:
            line = re.sub('\n', '', line)
            start_pos = line.find('<h>')
            end_pos = line.find('</h>') - len('</h>') + 1

            sent_cleaned = line.replace('<h>', '').replace('</h>', '')
            soft_skill_tokens = sent_cleaned[start_pos: end_pos].split()
            sent_tokens = nltk.word_tokenize(sent_cleaned)

            sent_cnt += 1
            n_token = 0
            for token in sent_tokens:
                sentences.append('Sentence: ' + str(sent_cnt))
                tokens.append(token)
                if token in soft_skill_tokens:
                    n_token += 1
                    if n_token == 1:
                        tags.append('B-hobby')
                    else:
                        tags.append('I-hobby')
                else:
                    tags.append('O')
    f.close()

print(len(sentences), len(tokens), len(tags))

# Working on profession text file
f = open('data/New-Profession-NER.txt', 'r')

lines = f.readlines()
lines = random.sample(lines, len(lines))

for line in lines:
    line = re.sub('\n', '', line)
    start_pos = line.find('<p>')
    end_pos = line.find('</p>') - len('</p>') + 1

    sent_cleaned = line.replace('<p>', '').replace('</p>', '')
    soft_skill_tokens = sent_cleaned[start_pos: end_pos].split()
    sent_tokens = nltk.word_tokenize(sent_cleaned)

    sent_cnt += 1

    n_token = 0
    for token in sent_tokens:
        sentences.append('Sentence: ' + str(sent_cnt))
        tokens.append(token)
        if token in soft_skill_tokens:
            n_token += 1
            if n_token == 1:
                tags.append('B-profession')
            else:
                tags.append('I-profession')
        else:
            tags.append('O')
f.close()

print(len(sentences), len(tokens), len(tags))
print("Total Number of sentences: {}".format(sent_cnt))

df = pd.DataFrame()
df['Sentences'] = sentences
df['tokens'] = tokens
df['tags'] = tags

df.to_csv("NER.csv", index=False)