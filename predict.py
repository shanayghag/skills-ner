import spacy
import re
import pickle
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForTokenClassification

ner = spacy.load('en_core_web_lg')

# Loading the pickle file
f = open("data/Multi-sentence Cached data/idx2tag.pb", 'rb')
idx2tag = pickle.load(f)

# load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("trained_model/", do_lower_case=False)

# load model
model = BertForTokenClassification.from_pretrained("trained_model/", num_labels=len(idx2tag))
device = "cuda"
model.to(device)

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(r" +", ' ', text)
    return text

def predictTags(text):
    text = clean_text(text)
    tokenized_sentence = tokenizer.encode(text)
    input_ids = torch.tensor([tokenized_sentence]).cuda()

    # predict
    model.eval();
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_indices = np.argmax(logits[0], axis=-1)

    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(idx2tag[label_idx])
            new_tokens.append(token)

    print(new_tokens)
    print(new_labels)

    i = 0
    extracted = set()
    temp_skill = []
    counter = 0
    prev_tag = ''
    for idx, tup in enumerate(zip(new_tokens, new_labels)):
        token, label = tup

        if label == 'O':
            counter += 1

        if counter == (len(new_tokens) - 2):
            tup = ('', '')
            extracted.add(tup)
            break

        elif 'skill' in label:
            if i == 0:
                temp_skill.append(token)
            else:
                if label.split('-')[0] == 'B':
                    extracted.add((' '.join(temp_skill), prev_tag.split('-')[1]))
                    temp_skill = []
                    temp_skill.append(token)

                elif prev_tag.split('-')[1] != label.split('-')[1]:
                    extracted.add((' '.join(temp_skill), prev_tag.split('-')[1]))
                    temp_skill = []
                    temp_skill.append(token)

                else:
                    temp_skill.append(token)


            prev_tag = label
            i += 1

        elif 'profession' in label:
            if i == 0:
                temp_skill.append(token)
            else:
                if label.split('-')[0] == 'B':
                    extracted.add((' '.join(temp_skill), prev_tag.split('-')[1]))
                    temp_skill = []
                    temp_skill.append(token)

                elif prev_tag.split('-')[1] != label.split('-')[1]:
                    extracted.add((' '.join(temp_skill), prev_tag.split('-')[1]))
                    temp_skill = []
                    temp_skill.append(token)

                else:
                    temp_skill.append(token)

            prev_tag = label
            i += 1

        elif 'hobby' in label:
            if i == 0:
                temp_skill.append(token)
            else:
                if label.split('-')[0] == 'B':
                    extracted.add((' '.join(temp_skill), prev_tag.split('-')[1]))
                    temp_skill = []
                    temp_skill.append(token)

                elif prev_tag.split('-')[1] != label.split('-')[1]:
                    extracted.add((' '.join(temp_skill), prev_tag.split('-')[1]))
                    temp_skill = []
                    temp_skill.append(token)

                else:
                    temp_skill.append(token)

            prev_tag = label
            i += 1

        elif ' '.join(temp_skill) not in extracted and idx == len(new_tokens) - 1 and prev_tag!= '':
            extracted.add((' '.join(temp_skill), prev_tag.split('-')[1]))

    return extracted

def dump2dict(results, text):
    softskills = []
    professions = []
    hobbies = []
    for result in results:
        entity, tag = result
        if tag == 'softskill':
            softskills.append(entity)

        if tag == 'hobby':
            hobbies.append(entity)

        if tag == 'profession':
            professions.append(entity)

    results_dict = {}
    results_dict['soft-skills'] = softskills
    results_dict['professions'] = professions
    results_dict['hobbies'] = hobbies

    text = clean_text(text)
    doc = ner(text)
    for ent in doc.ents:
        results_dict[str(ent.label_)] = ent.text

    return results_dict

if __name__ == '__main__':

    test_text = '''Thank You for your 5 Star ⭐️⭐️⭐️⭐️⭐️ reviews! REX, Hardrocker, Teen Idol, Superhero, PrimeTime/DayTime, Broadway and Movie Star, You Take My Breath Away, Sooner Or Later, Pirates Of Penzance, Grease, Scarlet Pimpernel, Street Hawk, Original DareDevil, Host of Solid Gold, As The World Turns, ”Billy Mack” on ”Love Actually Live”....now its time for a *SING OUT* TO YOU! Please clearly include your names (and any special pronunciations) if you want to specifically get mentioned, who it’s from, who it’s for...Cheers! ***No endorsement requests, thank you.'''

    results = predictTags(test_text)
    results = dump2dict(results, test_text)
    print(results)



