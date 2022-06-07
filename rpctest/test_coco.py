
import os
import pickle
import time
import sys

import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from cont import find_best_config
from vocabulary import Vocabulary
from model import EncoderCNN, DecoderRNN


def train_fn(model_file, train_loader, config):
    embed_size = 256        # dimensionality of image and word embeddings
    hidden_size = 512       # number of features in hidden state of the RNN decoder
    vocab_size = config["vocab_size"] #TODO: how to pass vocab_size??
    learning_rate = config["learning_rate"]
    
    total_step = math.ceil(len(train_loader.dataset) / 128)
    start_step=1 
    start_loss=0.0
    total_loss = start_loss
    
    
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Define the loss function
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Specify the learnable parameters of the model
    params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

    # Define the optimizer
    optimizer = torch.optim.Adam(params=params, lr=learning_rate)
    
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    encoder.train()
    decoder.train()

    # Keep track of train loss
    total_loss = start_loss

    # Start time for every 100 steps
    start_train_time = time.time()
    i_step = 1
    for batch in train_loader:
        images, captions = batch[0], batch[1]
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
        stats = "Train step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f" \
                    % (i_step, total_step, time.time() - start_train_time,
                       loss.item(), np.exp(loss.item()))
        print("\r" + stats, end="")
        sys.stdout.flush()
        i_step += 1
    
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "total_loss": total_loss
               }, model_file)
    print("final loss:" + str(total_loss / total_step))


def valid_fn(model_file, val_loader):
    """Validate the model for one epoch using the provided parameters. 
    Return the epoch's average validation loss and Bleu-4 score."""
    print("doing some validation")


ip0 = "http://localhost:7777"
ip1 = "http://localhost:7778"

shard1_path = "/users/vik1497/mop-test/coco/train1/"
shard2_path = "/users/vik1497/mop-test/coco/train2/"

val_path = "/users/vik1497/mop-test/coco/val/"

train_partitions = [shard1_path, shard2_path]

valid_partitions = [val_path]

worker_ips = [ip0, ip1]

vocab_threshold = 5
vocab = Vocabulary(vocab_threshold, vocab_from_file=True) # loading from file (change to annotations_file attribute?)
vocab_size = len(vocab)


param_grid = {
        'learning_rate': [1e-1, 1e-2],
        'vocab_size': [vocab_size]
}

param_names = [x for x in param_grid.keys()]

def find_combinations(combinations, p, i):
    """
    :param combinations:
    :param p:
    :param i:
    """
    if i < len(param_names):
        for x in param_grid[param_names[i]]:
            p[param_names[i]] = x
            find_combinations(combinations, p, i + 1)
    else:
        combinations.append(p.copy())

# Grid search. Creating all parameter configuration value sets.
train_configs = []
find_combinations(train_configs, {}, 0)

nepochs=2



find_best_config(nepochs, worker_ips, train_partitions, valid_partitions, train_fn, valid_fn, train_configs, preload_data_to_mem=True)