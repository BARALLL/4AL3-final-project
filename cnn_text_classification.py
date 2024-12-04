from datasets import load_dataset
from transformers import BertTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset, random_split, DataLoader, Dataset

import pickle
import os
import time
from tqdm import tqdm
import pathlib
import config

PATH = pathlib.Path(__file__).parent

torch.manual_seed(config.SEED)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# use pickle to store the tokenized version of the dataset because it is long to process each time otherwise
tokenized_datasets_save_path = PATH / config.PICKLE_FILE
if (tokenized_datasets_save_path).exists():
    # load from pickle file if it exists
    print(f"Loading tokenized dataset from {tokenized_datasets_save_path}...")
    with open(tokenized_datasets_save_path, "rb") as f:
        tokenized_datasets = pickle.load(f)
    
    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL)
    vocab = tokenizer.vocab
    
else:
    # load from Hugging Face datasets and tokenize
    print("Loading and tokenizing dataset from Hugging Face...")
    dataset = load_dataset(config.DATASET_NAME)

    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL)
    vocab = tokenizer.vocab

    tokenized_datasets = dataset.map(tokenize_function, batched=True, )

    # Save the tokenized dataset using pickle
    print(f"Saving tokenized dataset to {tokenized_datasets_save_path}...")
    with open(tokenized_datasets_save_path, "wb") as f:
        pickle.dump(tokenized_datasets, f)


tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


combined_dataset = ConcatDataset([tokenized_datasets['train'], tokenized_datasets['test']])
print(len(combined_dataset))
train_dataset, val_dataset, test_dataset = random_split(
    combined_dataset, config.SPLIT_PERCENTAGES, generator=torch.Generator().manual_seed(config.SEED)
)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print(
    "Training Set Size:",
    len(train_loader) * config.BATCH_SIZE,
    "Validation Set Size:",
    len(val_loader) * config.BATCH_SIZE,
    "Test Set Size:",
    len(test_loader) * config.BATCH_SIZE,
)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_0 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=n_filters,
                                kernel_size=filter_sizes[0])
        self.conv_1 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=n_filters,
                                kernel_size=filter_sizes[1])
        self.conv_2 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=n_filters,
                                kernel_size=filter_sizes[2])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)

        conved_0 = nn.functional.relu(self.conv_0(embedded))
        conved_1 = nn.functional.relu(self.conv_1(embedded))
        conved_2 = nn.functional.relu(self.conv_2(embedded))

        pooled_0 = nn.functional.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = nn.functional.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = nn.functional.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        return self.fc(cat)


INPUT_DIM = len(vocab)
OUTPUT_DIM = 1

model = CNN(INPUT_DIM, config.EMBEDDING_DIM, config.N_FILTERS, config.FILTER_SIZES, OUTPUT_DIM, config.DROPOUT)

id2label = lambda pred: "positive" if pred >  0.5 else "negative"

optimizer = optim.Adam(model.parameters(), weight_decay=config.L2_REG, lr=config.LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in tqdm(iterator, desc='Train', leave=False, unit=' samples', unit_scale=config.BATCH_SIZE):
        optimizer.zero_grad()
        text = batch['input_ids'].to(device)
        labels = batch['label'].unsqueeze(1).float().to(device).squeeze(1)
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(torch.sigmoid(predictions), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc='Eval', unit=' samples', unit_scale=config.BATCH_SIZE):
            text = batch['input_ids'].to(device)
            labels = batch['label'].unsqueeze(1).float().to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels.squeeze(1))
            acc = binary_accuracy(torch.sigmoid(predictions), labels.squeeze(1))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def predict(model, tokenizer, text, device):
    model.eval()
    tokenized_text = tokenizer(text, padding='max_length', truncation=True, max_length=config.MAX_LENGTH, return_tensors='pt')
    input_ids = tokenized_text['input_ids'].to(device)

    with torch.no_grad():
        prediction = torch.sigmoid(model(input_ids)).item()

    return prediction

def predict_sentiment(model, tokenizer, text, device):
    pred = predict(model, tokenizer, text, device)
    print(text)
    print("\t Sentiment:", id2label(pred), '\t', "Confidence score", pred)

best_valid_loss = float('inf')

if config.TRAIN:
    for epoch in tqdm(range(config.EPOCHS), desc='Epochs', leave=False, unit='epoch'):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), PATH / config.MODEL_FILE)
        tqdm.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        tqdm.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        tqdm.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    print('Finished Training')
else:
    print("No train so load model")

# Load the best model
model_path = PATH / config.MODEL_FILE
if pathlib.Path(model_path).exists():
    model.load_state_dict(torch.load(model_path, weights_only=True))
else:
    print(f"Could not find model at {model_path}. Please set TRAIN to True in the config file to train the model first")
    exit()

examples = [
    "This movie was absolutely amazing! I loved every minute of it.",
    "The flower field was pretty.",
    "The flower field was pretty!",

    # this is not a proper input, since the model was trained with only positive or negative sentence, not neutral ones
    "This is a completely neutral sentence right?", 
    
    # if we truely understand the context the sentence below should be classified as positive
    "This movie was absolutely garbage! In a good way. I loved every minute of it. 100% recommand", 
    
    "Meh.",
    "This movie was pretty time-wasting.",
    "The villain was bland.",
    "This movie was terrible. I hated it."
]

# for text in examples:
#     predicted_sentiment_negative = predict_sentiment(model, tokenizer, text, device)

test_loss, test_acc = evaluate(model, test_loader, criterion)
print("Test Loss:", test_loss, "Test Accuracy:", test_acc)