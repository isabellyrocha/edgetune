from tuning import InferenceServer
#from workloads.cifar10resnet.Resnet import Resnet
from keras.datasets import cifar10
from tensorflow import keras
from pathlib import Path
from utils import utils
import workloads.ic.lib.rapl.rapl as rapl
import tensorflow as tf
from ray import tune
from threading import Thread
import numpy as np
import json
import shutil
import time
import os
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import torch


class TextClassificationModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_class):
            super(TextClassificationModel, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
            self.fc = nn.Linear(embed_dim, num_class)
            self.init_weights()

        def init_weights(self):
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()

        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            return self.fc(embedded)

class MultiBudgetTraining(tune.Trainable):
    def setup(self, config):
        self.steps = 0
        self.epochs = 0
        self.inference_duration = None
        self.inference_energy = None
        self.inference_batch = None
        self.inference_cores = None

    def get_percentage(self, step):
        return np.minimum(1, step*0.1)


    #train_iter = AG_NEWS(split='train')

    def yield_tokens(self):
        tokenizer = get_tokenizer('basic_english')
        train_iter = AG_NEWS(split='train')
        for _, text in AG_NEWS(split='train'):
            yield tokenizer(text)

    ##text_pipeline = lambda x: vocab(tokenizer(x))
    ##label_pipeline = lambda x: int(x) - 1

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_batch(self, batch):
        vocab = build_vocab_from_iterator(self.yield_tokens(), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        tokenizer = get_tokenizer('basic_english')
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x) - 1
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
             label_list.append(label_pipeline(_label))
             processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
             text_list.append(processed_text)
             offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return label_list.to(device), text_list.to(device), offsets.to(device)

    #dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
    
    #train_iter = AG_NEWS(split='train')
    #num_class = len(set([label for (label, text) in train_iter]))

    def train_workload(self, model, dataloader):

        optimizer = torch.optim.SGD(model.parameters(), lr=5)

        model.train()
        total_acc, total_count = 0, 0
        start_time = time.time()
        print("Train worklaod method")
        for idx, (label, text, offsets) in enumerate(dataloader):
            print(idx)
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(predicted_label, label)
            print("Done with forward...")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            print("Done with backward...")
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                  total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(model, dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count

    def step(self):
        self.steps += 1

        ##### Setting training configurations #####
        LR = self.config.get("lr", 5)
        BATCH_SIZE = self.config.get("train_batch", 64)
        utils.set_training_cores(self.config.get("train_cores", 4))
        emsize = self.config.get("embed_dim", 64)

        train_iter = AG_NEWS(split='train')
        vocab = build_vocab_from_iterator(self.yield_tokens(), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        vocab_size = len(vocab)
        train_iter = AG_NEWS(split='train')
        num_class = len(set([label for (label, text) in train_iter]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

        #### Inference ###
        inf_serv_results = {}
        inf_serv_results = InferenceServer.runSearch(emsize, inf_serv_results) 
        self.inference_duration = inf_serv_results['inference_duration']
        self.inference_energy = inf_serv_results['inference_energy']
        self.inference_cores = inf_serv_results['config']['inference_cores']
        self.inference_batch = inf_serv_results['config']['inference_batch']
        

        epochs = self.steps * 2
        self.epochs += epochs

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        total_accu = None
        train_iter, test_iter = AG_NEWS()
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        num_train = int(len(train_dataset) * 0.2)
        split_train_, split_valid_ = \
            random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=self.collate_batch)
        #valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
        #                     shuffle=True, collate_fn=self.collate_batch)
        #test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
        #                     shuffle=True, collate_fn=self.collate_batch)
        
        percentage = self.get_percentage(self.epochs)
        total_images = len(train_dataloader) * percentage

        print("Starting training....")

        training_start = time.time()
        start_energy = rapl.RAPLMonitor.sample()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            optimizer = torch.optim.SGD(model.parameters(), lr=5)

            model.train()
            total_acc, total_count = 0, 0
            start_time = time.time()
            #print("Train worklaod method")
            #print("Size: %d" % len(train_dataloader))
            for idx, (label, text, offsets) in enumerate(train_dataloader):
                #print(idx)
                optimizer.zero_grad()
                predicted_label = model(text, offsets)
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(predicted_label, label)
                #print("Done with forward...")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                #print("Done with backward...")
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
                print(total_count)
                #if True: #idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                if idx % 50 == 0:
                    print('| epoch {:3d} | {:5d}/{:5d} batches '
                          '| accuracy {:8.3f}'.format(epoch, idx, len(train_dataloader),
                                                  total_acc/total_count))
                #total_acc, total_count = 0, 0
                start_time = time.time()
                if idx >= total_images:
                    break

            #accu_val = evaluate(model, valid_dataloader)
            #print("Evaluation..")
            #if total_accu is not None and total_accu > accu_val:
            #    scheduler.step()
            #else:
            #    total_accu = accu_val
            #if idx >= total_images:
            #    break

        training_duration = time.time() - training_start
        end_energy = rapl.RAPLMonitor.sample()
        diff = end_energy-start_energy
        training_energy = diff.energy('package-0')
        training_accuracy = total_acc/total_count # total_accu #float(train_acc_metric.result())

        #model.save(directory_name)
        if training_accuracy is not None:
            print(training_duration)
            print(self.inference_duration)
            print(training_accuracy)
            runtime_ratio = (training_duration*self.inference_duration)/training_accuracy
        else:
            runtime_ratio = 0

        result = {
            "epochs": self.epochs,
            "runtime_ratio": runtime_ratio,
            "training_accuracy": training_accuracy,
            "training_duration": training_duration,
            "training_energy": training_energy,
            "inference_duration": self.inference_duration,
            "inference_energy": self.inference_energy,
            "inference_cores": self.inference_cores,
            "inference_batch": self.inference_batch
        }

        return result

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({
                "epochs": self.epochs,
                "inference_duration": self.inference_duration,
                "inference_energy": self.inference_energy,
                "inference_batch": self.inference_batch,
                "inference_cores": self.inference_cores
            }))
        return path

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path) as f:
            j = json.loads(f.read())
            self.epochs = j["epochs"]
            self.inference_duration = j["inference_duration"]
            self.inference_energy = j["inference_energy"]
            self.inference_batch = j["inference_batch"]
            self.inference_cores = j["inference_cores"]
