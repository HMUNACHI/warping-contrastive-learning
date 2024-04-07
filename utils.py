import torch
import numpy as np
import pandas as pd
import gzip, csv, copy

from typing import List, Dict
from torch.optim import AdamW
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

torch.manual_seed(0)
np.random.seed(0)


def load_sts_dataset(file_name: str) -> Dict[str, List[List[str]]]:
    """
    Load and prepare the STS dataset from a compressed TSV file.

    Args:
        file_name (str): The path to the compressed TSV file.

    Returns:
        Dict[str, List[List[str]]]: A dictionary containing STS dataset splits with sentence pairs and scores.
    """
    sts_samples = {'test': [[],[],[]]}

    with gzip.open(file_name, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            if row['split'] == 'test':
                sts_samples['test'][0].append(str(row['sentence1']))
                sts_samples['test'][1].append(str(row['sentence2']))
                sts_samples['test'][2].append(float(row['score']))

    return sts_samples


def tokenize_sentence_pair_dataset(dataset: List[str],
                                   tokenizer: BertTokenizer,
                                   max_length: int = 512) -> torch.utils.data.TensorDataset:
    """
    Tokenize a dataset of sentence pairs using a given tokenizer and prepare it as a TensorDataset.

    Args:
        dataset (List[str]): List containing sentence pairs and labels.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        max_length (int, optional): Maximum sequence length for padding and truncation. Default is 512.

    Returns:
        TensorDataset: A TensorDataset containing tokenized input_ids, attention_mask, and labels.
    """
    tokenized_data1 = tokenizer(dataset[0],
                                return_tensors='pt',
                                padding='max_length',
                                max_length=max_length,
                                truncation=True)

    tokenized_data2 = tokenizer(dataset[1],
                                return_tensors='pt',
                                padding='max_length',
                                max_length=max_length,
                                truncation=True)

    if len(dataset) == 3:
        tokenized_dataset = torch.utils.data.TensorDataset(tokenized_data1['input_ids'],
                                                            tokenized_data1['attention_mask'],
                                                            tokenized_data2['input_ids'],
                                                            tokenized_data2['attention_mask'],
                                                            torch.tensor(dataset[2]))
    else:
        tokenized_dataset = torch.utils.data.TensorDataset(tokenized_data1['input_ids'],
                                                        tokenized_data1['attention_mask'],
                                                        tokenized_data2['input_ids'],
                                                        tokenized_data2['attention_mask'])

    return tokenized_dataset



def get_dataloader(tokenized_dataset: torch.utils.data.Dataset,
                   batch_size: int,
                   shuffle: bool = False) -> DataLoader:
    """
    Create and return a DataLoader for a tokenized dataset.

    Args:
        tokenized_dataset (Dataset): The tokenized dataset to create a DataLoader for.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data in the DataLoader. Default is False.

    Returns:
        DataLoader: A DataLoader for the tokenized dataset.
    """
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)



def cosine_sim(a, b):
    """
    Calculate cosine similarity between two matrices of vectors.

    Formula = dot (a,b) / |a| * |b|

    Args:
        a (torch.Tensor): First matrix of vectors (batch_size, vector_dim).
        b (torch.Tensor): Second matrix of vectors (batch_size, vector_dim).

    Returns:
        torch.Tensor: Pairwise cosine similarity matrix (batch_size, batch_size).

    """
    dot_product = torch.einsum('ij,kj->ik', a, b)
    norm_a = torch.sqrt(torch.einsum('ij,ij->i', a, a))
    norm_b = torch.sqrt(torch.einsum('ij,ij->i', b, b))
    norm_product = torch.einsum('i,j->ij', norm_a, norm_b)
    return dot_product / norm_product



def eval_loop(model: torch.nn.Module,
              eval_dataloader: DataLoader,
              device: str) -> List[float]:
    """
    Evaluate a model's performance using cosine similarity as the distance metric,
    and calculate Pearson and Spearman correlation coefficients.

    Args:
        model (torch.nn.Module): The pre-trained model to evaluate.
        eval_dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (str): Device to perform evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        EvaluationResult: A list containing Pearson and Spearman correlation coefficients.
    """
    model.to(device)
    model.eval()

    vectors_one = []
    vectors_two = []
    labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            embeddings1 = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device))[1]
            embeddings2 = model(input_ids=batch[2].to(device), attention_mask=batch[3].to(device))[1]
            vectors_one.append(embeddings1.cpu())
            vectors_two.append(embeddings2.cpu())
            labels.extend(batch[4].cpu().numpy())

    vectors_one = torch.cat(vectors_one)
    vectors_two = torch.cat(vectors_two)
    labels = np.array(labels)
    similarities = cosine_sim(vectors_one, vectors_two).numpy().diagonal()
    eval_pearson_cosine, _ = pearsonr(similarities, labels)
    eval_spearman_cosine, _ = spearmanr(similarities, labels)
    return [eval_pearson_cosine, eval_spearman_cosine]


def load_nli_paws_dataset(file_name: str) -> Dict[str, List[List[str]]]:
    """
    Load and prepare the STS dataset from a compressed TSV file.

    Args:
        file_name (str): The path to the compressed TSV file.

    Returns:
        Dict[str, List[List[str]]]: A dictionary containing NLI dataset splits with sentence pairs and scores.
    """
    label = {'entailment' : 1, 'contradiction' : 0, 'neutral' : 2}
    nli_samples = {'train': [[],[],[]], 'dev': [[],[],[]]}
    data = pd.read_csv(file_name, nrows=593841, compression='gzip', delimiter='\t').head(10000)

    for row in data[data["split"] == "train"].values:
        if row[-1] == 'entailment' :
            nli_samples['train'][0].append(str(row[3]))
            nli_samples['train'][1].append(str(row[4]))
            nli_samples['train'][2].append(label[row[5]])

    for row in data[data["split"] == "dev"].values:
        if row[-1] == 'entailment' :
            nli_samples['dev'][0].append(str(row[3]))
            nli_samples['dev'][1].append(str(row[4]))
            nli_samples['dev'][2].append(label[row[5]])


    data = pd.read_csv('data/qmnli/train.csv')[:1000]
    for row in data.values:
        if row[-1] == 1:
            nli_samples['train'][0].append(str(row[2]))
            nli_samples['train'][1].append(str(row[5]))
            nli_samples['train'][2].append(row[-1])

    data = pd.read_csv('data/qmnli/validation_matched.csv').head(1000)
    for row in data.values:
        if row[-1] == 1:
            nli_samples['dev'][0].append(str(row[2]))
            nli_samples['dev'][1].append(str(row[5]))
            nli_samples['dev'][2].append(row[-1])

    print("Some data samples from the NLI dataset:\n")
    for i in range(10):
        print("Sample", i+1)
        print("Sentence a:", nli_samples['train'][0][i])
        print("Sentence b:",nli_samples['train'][1][i])
        print("")

    return nli_samples


def train_loop(model, optimizer, train_dataloader, validation_dataloader, test_dataloader, num_epochs, device):
    """
    Train a model using a training loop and perform periodic validation.
    Args:
        model: The model to be trained.
        optimizer: The optimizer for model parameter updates.
        train_dataloader: DataLoader for the training dataset.
        validation_dataloader: DataLoader for the validation dataset.
        num_epochs: Number of training epochs.
        device: Device to perform training and validation (e.g., 'cuda' or 'cpu').
    Returns:
        Tuple[float, float]: Training and validation loss at the end of training.
    """
    model.to(device)
    best_model = copy.deepcopy(model)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()
            train_loss = model(
                sentence_1_ids=batch[0].to(device),
                sentence_1_mask=batch[1].to(device),
                sentence_2_ids=batch[2].to(device),
                sentence_2_mask=batch[3].to(device),
                labels=batch[4].to(device)
            )
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in validation_dataloader:
                val_loss = model(
                    sentence_1_ids=batch[0].to(device),
                    sentence_1_mask=batch[1].to(device),
                    sentence_2_ids=batch[2].to(device),
                    sentence_2_mask=batch[3].to(device),
                    labels=batch[4].to(device)
                )
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(validation_dataloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)

        results = eval_loop(model.bert, test_dataloader, device)
        print(f'\nPearson correlation: {results[0]:.2f}\nSpearman correlation: {results[1]:.2f}')
        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}")
        
    return best_model


def train_and_evaluate(model, tokenizer, test_dataloader, num_epochs=3, batch_size=8, maxlen=128):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bert_path = 'weights/bert_tiny.bin'
    nli_dataset = load_nli_paws_dataset('data/AllNLI.tsv.gz')
    print('Train:', len(nli_dataset['train'][0]))
    print('Validation:', len(nli_dataset['dev'][0]))

    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'], tokenizer, max_length=maxlen)
    tokenized_val = tokenize_sentence_pair_dataset(nli_dataset['dev'], tokenizer, max_length=maxlen)
    train_dataloader = get_dataloader(tokenized_train, batch_size=batch_size, shuffle=True)
    validation_dataloader = get_dataloader(tokenized_val, batch_size=batch_size, shuffle=True)

    bert = model({'hidden_size': 128, 'num_attention_heads': 2, 'num_hidden_layers': 2, 
                             'intermediate_size': 512, 'vocab_size': 30522}, bert_path)
    print(sum(p.numel() for p in bert.parameters() if p.requires_grad))

    optimizer = AdamW(bert.parameters(), lr=5e-5)
    return train_loop(bert, optimizer, train_dataloader, validation_dataloader, test_dataloader, num_epochs, device)