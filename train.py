import pickle
import torch
import torch
import os
import numpy as np
import torch.nn.functional as F
from tqdm import trange
from tqdm import tqdm

from torch.optim import Adam
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report, accuracy_score, f1_score

from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

f = open("data/Multi-sentence Cached data/train_tags.pb", 'rb')
train_tags = pickle.load(f)
f.close()

f = open("data/Multi-sentence Cached data/test_tags.pb", 'rb')
test_tags = pickle.load(f)
f.close()

f = open("data/Multi-sentence Cached data/train_inputs.pb", 'rb')
train_inputs = pickle.load(f)
f.close()

f = open("data/Multi-sentence Cached data/test_inputs.pb", 'rb')
test_inputs = pickle.load(f)
f.close()

f = open("data/Multi-sentence Cached data/train_attn_masks.pb", 'rb')
train_attn_masks = pickle.load(f)
f.close()

f = open("data/Multi-sentence Cached data/test_attn_masks.pb", 'rb')
test_attn_masks = pickle.load(f)
f.close()

# convert all to tensors
train_inputs = torch.tensor(train_inputs)
train_attn_masks = torch.tensor(train_attn_masks)
train_tags = torch.tensor(train_tags)

test_inputs = torch.tensor(test_inputs)
test_attn_masks = torch.tensor(test_attn_masks)
test_tags = torch.tensor(test_tags)

print('train_inputs tensor:', train_inputs.shape)
print('train_attn_masks tensor:', train_attn_masks.shape)
print('train_tags tensor:', train_tags.shape)
print('\n')
print('test_inputs tensor:', test_inputs.shape)
print('test_attn_masks tensor:', test_attn_masks.shape)
print('test_tags tensor:', test_tags.shape)

BATCH_SIZE = 32

# only set token embedding, tag_embedding ,attention embedding, NO segment embedding
train_data = TensorDataset(train_inputs, train_attn_masks, train_tags)
train_sampler = RandomSampler(train_data)

# drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=True)

test_data = TensorDataset(test_inputs, test_attn_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, sampler=test_sampler,)

device = "cuda"

f = open("data/Multi-sentence Cached data/idx2tag.pb", 'rb')
idx2tag = pickle.load(f)
f.close()

print(idx2tag)

model = BertForTokenClassification.from_pretrained('bert-base-cased',
                                                   num_labels=len(idx2tag),
                                                   output_attentions = False,
                                                   output_hidden_states = False,)

model.to(device)

FULL_FINETUNING = True

if FULL_FINETUNING:
    # fine tune all layer parameters of the model
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    # only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

# Set epoch and max_grad_norm (for gradient clipping)
epochs = 5
max_grad_norm = 1.0

# Cacluate train optimization num
num_train_optimization_steps = int(np.ceil(len(train_inputs) / BATCH_SIZE)) * epochs

# Set epoch and max_grad_norm (for gradient clipping)
epochs = 5
max_grad_norm = 1.0

# Cacluate train optimization num
num_train_optimization_steps = int(np.ceil(len(train_inputs) / BATCH_SIZE)) * epochs

train_loss_values, val_loss_values = [], []

for epoch in range(epochs):
    # ========================================
    #               Training
    # ========================================
    model.train()

    train_loss = 0
    for batch in tqdm(train_dataloader, desc='Epoch ' + str(epoch)):
        batch = tuple(t.to(device) for t in batch)
        batch_inputs, batch_attn_masks, batch_tags = batch

        batch_tags = batch_tags.type(torch.LongTensor).to(device)
        batch_inputs = batch_inputs.type(torch.LongTensor).to(device)
        batch_attn_masks = batch_attn_masks.type(torch.LongTensor).to(device)

        # always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()

        # forward pass
        outputs = model(batch_inputs, token_type_ids=None, attention_mask=batch_attn_masks, labels=batch_tags)
        loss = outputs[0]

        # backward pass
        loss.backward()

        # track train loss
        train_loss += loss.item()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        # update parameters
        optimizer.step()
        optimizer.zero_grad()

    # print average train_loss
    train_loss = train_loss / len(train_dataloader)
    train_loss_values.append(train_loss)
    print('Average train loss:', train_loss)

    # ========================================
    #               Validation
    # ========================================
    model.eval();

    eval_loss, eval_accuracy = 0, 0
    predictions, true_labels = [], []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        batch_inputs, batch_attn_masks, batch_tags = batch

        batch_tags = batch_tags.type(torch.LongTensor).to(device)
        batch_inputs = batch_inputs.type(torch.LongTensor).to(device)
        batch_attn_masks = batch_attn_masks.type(torch.LongTensor).to(device)

        with torch.no_grad():
            outputs = model(batch_inputs,
                            token_type_ids=None,
                            attention_mask=batch_attn_masks,
                            labels=batch_tags)

        # CALC VAL LOSS
        eval_loss += outputs[0].mean().item()

        # CALC VAL ACC
        # get pred labels
        logits = outputs[1]
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()

        # get true labels
        label_ids = batch_tags.to('cpu').numpy()

        # Only predict the real word, mark=0, will not calculate
        batch_attn_masks = batch_attn_masks.to('cpu').numpy()

        for i, mask in enumerate(batch_attn_masks):
            # Real one
            temp_1 = []
            # Predict one
            temp_2 = []

            for j, m in enumerate(mask):
                # Mark=0, meaning its a pad word, dont compare
                if m:
                    if idx2tag[label_ids[i][j]] != "X" and idx2tag[label_ids[i][j]] != "[CLS]" and idx2tag[
                        label_ids[i][j]] != "[SEP]":  # Exclude the X label
                        temp_1.append(idx2tag[label_ids[i][j]])
                        temp_2.append(idx2tag[logits[i][j]])
                else:
                    break
            true_labels.append(temp_1)
            predictions.append(temp_2)

    eval_loss = eval_loss / len(test_dataloader)
    val_loss_values.append(eval_loss)
    print('Validation loss:', eval_loss)
    print('Validation accuracy:', accuracy_score(true_labels, predictions))
    print('Validation f1 score:', f1_score(true_labels, predictions))

# Save a trained model, configuration and tokenizer
model_to_save = model.module if hasattr(model, 'module') else model

model_dirname = 'trained_model'
if not os.path.exists(model_dirname):
    os.makedirs(model_dirname)

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(model_dirname, "multi-ner-model.bin")
output_config_file = os.path.join(model_dirname, "config.json")

# Save model into file
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(model_dirname)

report = classification_report(true_labels, predictions, digits=4)
print('Clasification report:\n', report)

