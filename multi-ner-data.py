import csv
import random
import re

from nltk import sent_tokenize, word_tokenize
import pandas as pd
from tqdm import tqdm

def find_all_indexes(input_str, search_str):
    '''
    Function to find the index of a sub-string
    :param input_str: str()
    :param search_str: str()
    :return: list()
    '''
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

def merge_different_hobby_data(text_file_path, data_file_paths, merged_data_path):
    '''
    Function to merge different hobbies together for creating training for NER.
    :param text_file_path: Path of the file you want to merge
    :param data_file_paths: List of file paths which you want to merge with
    :param merged_data_path: Path of the file where you want to stored the merged data
    :return: None
    '''
    hobby_sentences = []

    for file in data_file_paths:
        data_file = open(file, 'r')
        lines = data_file.readlines()
        lines = random.sample(lines, len(lines))

        for line in lines:
            line = re.sub("\n", '', line)
            hobby_sentences.append(line)

    hobby_sentences = random.sample(hobby_sentences, len(hobby_sentences))
    choices_list = [3, 4, 5, 6, 7]

    text_file = open(text_file_path, 'r')
    text_lines = text_file.readlines()
    text_lines = random.sample(text_lines, len(text_lines))

    merged_file = open(merged_data_path, 'w')

    for text in tqdm(text_lines):

        text = re.sub("\n", '', text) + '. '

        rand_choice_idx = random.randint(0, len(choices_list) - 1)
        counter = choices_list[rand_choice_idx]

        temp_str = ""

        for i in range(0, counter):

            rand_idx = random.randint(0, len(hobby_sentences) - 1)
            temp_str += hobby_sentences[rand_idx] + '. '

        temp_str = text + temp_str + '\n'
        merged_file.write(temp_str)

def merge_text_files(merge_files_path, save_file_path):
    '''
    Function to merge two text files.
    :param merge_files_path: list of the files you want to merge
    :param save_file_path: path to save the merged file
    :return: None
    '''
    temp_sentence_list = []

    for file in tqdm(merge_files_path):
        f = open(file, 'r')
        lines = f.readlines()

        for line in tqdm(lines):
            line = re.sub('\n', '', line)
            temp_sentence_list.append(line + '\n')

    save_file = open(save_file_path, 'w')
    for sent in tqdm(temp_sentence_list):
        save_file.write(sent)

def merge_lines_from_textfile(file_path, save_file_path, split_ratio):
    '''
    Merge line from same text file by splitting the file into two lists and randomly selecting a random number of lines from the second list.
    :param file_path: path of the file whose line you want to merge
    :param save_file_path: path where you want to save the file
    :param split_ratio: ratio by which you want to split the file into two list which are further used for data generation.
    :return: None
    '''
    file = open(file_path, 'r')
    file_lines = file.readlines()
    file_length = len(file_lines)
    file_split_idx = file_length * split_ratio
    text_list1 = file_lines[0:int(file_split_idx)]
    text_list2 = file_lines[int(file_split_idx):]

    choices_list = [3, 4, 5, 6, 7, 8]
    sent_merged_list = []
    save_file = open(save_file_path, 'w')

    for text in text_list1:
        text = re.sub('\n', '', text)
        temp_str = ''
        randChoiceIdx = random.randint(0, len(choices_list) - 1)
        counter = choices_list[randChoiceIdx]

        for i in range(0,counter):
            rand_idx = random.randint(0, len(text_list2) - 1)
            temp_str1 = text_list2[rand_idx]
            temp_str1 = re.sub('\n', '', temp_str1)
            temp_str += temp_str1 + '. '

        temp_str = text + '. ' + temp_str
        sent_merged_list.append(temp_str)
        temp_str += '\n'
        save_file.write(temp_str)


def create_iob_tags(text_file_path, sent_cnt, sentences, tokens, tags):
    '''
    Function to create IOB tags for each line in a text file.
    Assumption is that each line contains sentences having only one entity in it.
    :param text_file_path: path of the labeled data you want to work on
    :param sent_cnt: Sentence count
    :param sentences: list of senetnces
    :param tokens: list of tokens
    :param tags: list of IOB tags
    :return: sentences, tokens, tags, sent_cnt
    '''
    text_file = open(text_file_path, 'r')
    lines = text_file.readlines()

    for line in lines:
        sent_cnt += 1
        line = re.sub('\n', '', line)
        sent_tokens = sent_tokenize(line)
        for sent in sent_tokens:
            sent = re.sub('\n', '', sent)
            token_type_flag = 0

            if re.search(r'<p>', sent) != None:
                start_pos = sent.find('<p>')
                end_pos = sent.find('</p>') - len('</p>') + 1
                sent_cleaned = sent.replace('<p>', '').replace('</p>', '')
                token_type_flag = 1

            if re.search(r'<h>', sent) != None:
                start_pos = sent.find('<h>')
                end_pos = sent.find('</h>') - len('</h>') + 1
                sent_cleaned = sent.replace('<h>', '').replace('</h>', '')
                token_type_flag = 2

            if re.search(r'<b>', sent) != None:
                start_pos = sent.find('<b>')
                end_pos = sent.find('</b>') - len('</b>') + 1
                sent_cleaned = sent.replace('<b>', '').replace('</b>', '')
                token_type_flag = 3

            entity_tokens = sent_cleaned[start_pos: end_pos].split()
            new_sent_tokens = word_tokenize(sent_cleaned)

            n_token = 0
            for token in new_sent_tokens:
                sentences.append('Sentence: ' + str(sent_cnt))
                tokens.append(token)
                if token in entity_tokens:
                    n_token += 1

                    if n_token == 1:

                        if token_type_flag == 1:
                            tags.append('B-profession')

                        elif token_type_flag == 2:
                            tags.append("B-hobby")

                        elif token_type_flag == 3:
                            tags.append("B-softskill")

                    else:
                        if token_type_flag == 1:
                            tags.append('I-profession')

                        elif token_type_flag == 2:
                            tags.append("I-hobby")

                        elif token_type_flag == 2:
                            tags.append("I-softskill")
                else:
                    tags.append('O')
    return sentences, tokens, tags, sent_cnt

def get_iob_tags_from_inline_entities(lines_list, sent_cnt, tokens, tags, sentences):
    '''
    Function to create IOB tags for each line present in the list.
    Each line should have sentences which contain one or more than one entities in it.
    :param lines_list: list of lines
    :param sent_cnt: Sentence count
    :param tokens: list of tokens
    :param tags: list of tags
    :param sentences: list of sentences
    :return: sentences, tokens, tags, sent_cnt
    '''
    replace_pattern = re.compile(r'<p>|</p>|<h>|</h>|<b>|</b>')
    pattern = re.compile(r'<p>[a-zA-Z\s]+</p>|<h>[a-zA-Z\s]+</h>|<b>[a-zA-Z\s]+</b>')

    for line in tqdm(lines_list):
        sent_cnt += 1
        entity_matches = re.findall(pattern, line)

        entity_tags = {}

        line = re.sub(replace_pattern, '', line)
        entity_list = []

        for match in entity_matches:

            if re.search(r'<p>', match) != None:
                match = re.sub(replace_pattern, '', match)
                match_splits = match.split()

                match_counter = 0

                for split in match_splits:
                    entity_list.append(split)
                    if match_counter == 0:
                        entity_tags[split] = 'B-profession'

                    else:
                        entity_tags[split] = 'I-profession'
                    match_counter += 1

            if re.search(r'<h>', match) != None:
                match = re.sub(replace_pattern, '', match)
                match_splits = match.split()

                match_counter = 0

                for split in match_splits:
                    entity_list.append(split)
                    if match_counter == 0:
                        entity_tags[split] = 'B-Hobby'

                    else:
                        entity_tags[split] = 'I-Hobby'
                    match_counter += 1

            if re.search(r'<b>', match) != None:
                match = re.sub(replace_pattern, '', match)
                match_splits = match.split()

                match_counter = 0

                for split in match_splits:
                    entity_list.append(split)
                    if match_counter == 0:
                        entity_tags[split] = 'B-softskill'

                    else:
                        entity_tags[split] = 'I-softskill'
                    match_counter += 1

        word_tokens = word_tokenize(line)

        for token in word_tokens:
            sentences.append('Sentence: ' + str(sent_cnt))
            tokens.append(token)

            if token in entity_list:
                tags.append(entity_tags[token])

            else:
                tags.append("O")

    return sentences, tokens, tags, sent_cnt


if __name__ == '__main__':

    # Creating multi-entity data for NER model

    # Merging hobbies data along with different hobbies data to built a multi-entity data
    text_file_path = "data/Data for NER/Hobbies-NER.txt"
    data_file_paths = [
        "data/Data for NER/Instrument-NER.txt",
        "data/Data for NER/Music-NER.txt",
        "data/Data for NER/New-Dance-NER.txt",
        "data/Data for NER/New-Sports-NER.txt"
    ]
    merged_data_path = "data/Data for NER/merged-NER.txt"
    merge_different_hobby_data(text_file_path, data_file_paths, merged_data_path)

    #Merging profession data with hobby data to built multi-entity data
    text_file_path = "data/New-Profession-NER.txt"
    data_file_paths = [
        "data/Data for NER/Instrument-NER.txt",
        "data/Data for NER/Music-NER.txt",
        "data/Data for NER/New-Dance-NER.txt",
        "data/Data for NER/New-Sports-NER.txt",
        "data/Data for NER/Hobbies-NER.txt"
    ]
    merged_data_path = "data/Data for NER/merged-profession-NER.txt"
    merge_different_hobby_data(text_file_path, data_file_paths, merged_data_path)

    # Merging two text files
    merge_files_path = [
        "data/Data for NER/merged-NER.txt",
        "data/Data for NER/merged-profession-NER.txt"
    ]
    save_file_path = "data/Data for NER/mix-NER-data.txt"
    merge_text_files(merge_files_path, save_file_path)

    # Creating multi-profession entity data
    file_path = "data/New-Profession-NER.txt"
    save_file_path = "data/Mixed-Profession-NER.txt"
    merge_lines_from_textfile(file_path, save_file_path)

    # Merging two text files
    merge_files_path = [
        "data/Mixed-Profession-NER.txt",
        "data/Data for NER/mix-NER-data.txt"
    ]
    save_file_path = "data/Data for NER/train-data.txt"
    merge_text_files(merge_files_path, save_file_path)

    # Load data and remove rows having columns less than 3
    with open('data/Data for text generation/job_test.csv', 'r') as f:
        csv_read = csv.reader(f)
        job_test = []
        for row in csv_read:
            if len(row) == 3:
                job_test.append(row)

    with open('data/Data for text generation/job_test.csv', 'w') as f:
        csv_write = csv.writer(f)
        for row in job_test:
            csv_write.writerow(row)

    # cleaning the 'job' dataframe
    job = pd.read_csv('data/Data for text generation/job_train.csv')
    job = job.append(pd.read_csv('data/Data for text generation/job_test.csv'))
    cv = pd.read_csv('data/Data for text generation/cv_test.csv')

    print("Length of job dataframe: {}".format(len(job)))

    cvList = cv['sentence'].to_list()
    isSoftSkill = cv['is_soft_skill'].to_list()

    cvFile = open("data/Data for NER/cv.txt", 'w')

    for cvSent, softskillFlag in zip(cvList, isSoftSkill):
        if softskillFlag >= 0.67:
            cvSent += '\n'
            cvFile.write(cvSent)
    cvFile.close()

    # Creating multi-softskill entity data
    file_path = "data/Data for NER/cv.txt"
    save_file_path = "data/Mixed-Softskills-NER.txt"
    merge_lines_from_textfile(file_path, save_file_path)

    # Merging two text files
    merge_files_path = [
        "data/Mixed-Profession-NER.txt",
        "data/Data for NER/train-data.txt"
    ]
    save_file_path = "data/Data for NER/train-ner.txt"
    merge_text_files(merge_files_path, save_file_path)

    # Creating IOB tags from job dataframe
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
            for word in word_tokenize(sent):
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
        sent_tokens = word_tokenize(sent_cleaned)
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

    # Create IOB tags from text data
    text_file_path = "data/Data for NER/train-ner.txt"
    sentences, tokens, tags, sent_cnt = create_iob_tags(text_file_path, sent_cnt, sentences, tokens, tags)

    # Create IOB tags for bio data
    df = pd.read_csv("data/Data for text generation/bio.csv", error_bad_lines=False)
    bio_list = df['bio'][:110]
    sentences, tokens, tags, _ = get_iob_tags_from_inline_entities(bio_list, sent_cnt, tokens, tags, sentences)

    # Create a dataframe.
    test_df = pd.DataFrame()
    test_df['sentences'] = sentences
    test_df['tokens'] = tokens
    test_df['tags'] = tags

    # Saving the dataframe into a csv file.
    test_df.to_csv("data/train.csv", index=False)
    print("Done!!!")





