import workloads.ic.lib.rapl.rapl as rapl
#from workloads.cifar10resnet.Resnet import Resnet
from keras.datasets import cifar10
from tensorflow import keras
from pathlib import Path
from utils import utils
import tensorflow as tf
from ray import tune
import json
import time
import os

class Inference(tune.Trainable):
    def setup(self, config):
        self.timestep = 0
    
    def step(self):
        self.timestep += 1

        ##### Setting training configurations #####
        emsize = self.config.get("embed_dim", 64)
        val_batch_size = self.config.get("inference_batch", 32)
        utils.set_inference_cores(self.config.get("inference_cores", 4))
        
        print("Getting training data..")
        train_iter = AG_NEWS(split='train')
        vocab = build_vocab_from_iterator(self.yield_tokens(), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        vocab_size = len(vocab)
        train_iter = AG_NEWS(split='train')
        num_class = len(set([label for (label, text) in train_iter]))

        #device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model =  TextClassificationModel(vocab_size, emsize, num_class)#.to(device)
        print("Model created..")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        total_accu = None
        train_iter, test_iter = AG_NEWS()
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        num_train = int(len(train_dataset) * 0.95)
        split_train_, split_valid_ = \
            random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        #train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
        #                      shuffle=True, collate_fn=self.collate_batch)
        #valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
        #                      shuffle=True, collate_fn=self.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size,
                             shuffle=True, collate_fn=self.collate_batch)

        print("Starting training....")

        inference_start = time.time()
        start_energy = rapl.RAPLMonitor.sample()

        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(test_dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
                break
        eval_acc = total_acc/total_count

        inference_duration = time.time() - inference_start
        end_energy = rapl.RAPLMonitor.sample()
        diff = end_energy-start_energy
        inference_energy = diff.energy('package-0')

        result = {
            "inference_duration": inference_duration,
            "inference_energy": inference_energy
        }

        return result

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]
