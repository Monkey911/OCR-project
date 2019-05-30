from DataIterator import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import torch
#from model import *
from attention_model import *
import time
import numpy as np


def create_vocab(annotations):
    vocab = {}
    f = open(annotations)
    index = 0

    lines = f.readlines()
    for line in lines:
        label = line.split("\t")[1].strip()
        for letter in label:
            if letter not in vocab:
                vocab[letter] = index
                index += 1
    f.close()
    vocab['<UNK>'] = index
    vocab['<BOS>'] = index + 1
    vocab['<EOS>'] = index + 2
    vocab['<PAD>'] = index + 3
    return vocab


def train(train_data_iterator, valid_data_iterator, model, criterion, optimizer):
    total_step = len(train_data_iterator)

    # Start Training
    for epoch in range(100):
        print("Training for epoch " + str(epoch) + ".")
        avg_train_loss = train_epoch(train_data_iterator, criterion, model, optimizer)
        print("Training Loss: " + str(avg_train_loss))
        avg_valid_loss = valid_epoch(valid_data_iterator, criterion, model)
        print("Validation Loss: " + str(avg_valid_loss))

        # Saving each epoch model
        chkpt_name = os.path.join("checkpoints/model_full" + str(epoch) + ".chkpt")
        torch.save(model.state_dict(), chkpt_name)


def train_epoch(train_data_iterator, criterion, model, optimizer):
    t = tqdm(train_data_iterator, mininterval=1, desc='-(Training)', leave=False)
    total_loss = 0
    cntr = 0
    for batch in t:
        _, images, labels = batch
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        gold = labels.contiguous().view(-1)
        model.train()
        model.zero_grad()
        _, pred = model(images, labels)

        loss = criterion(pred, gold)
        loss.backward()
        optimizer.step()
        description = "Loss: " + str(loss.item())
        t.set_description(description)
        cntr += 1
        total_loss += loss.item()

    avg_loss = total_loss / float(cntr)
    return avg_loss


def valid_epoch(valid_data_iterator, criterion, model):
    model.eval()
    t = tqdm(valid_data_iterator, mininterval=1, desc='-(Validation)', leave=False)
    total_loss = 0
    cntr = 0
    accuracies = []
    for batch in t:
        images, labels = batch
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        gold = labels.contiguous().view(-1)
        pred, fpred = model(images, labels)
        predictions = pred.argmax(2).contiguous().view(-1)
        accuracies.append((predictions == gold).sum().item() / gold.size(0))
        loss = criterion(fpred, gold)
        description = "Loss: " + str(loss.item())
        t.set_description(description)
        cntr += 1
        total_loss += loss.item()

    print("average accuracy was " + str(np.mean(accuracies)))
    avg_loss = total_loss / float(cntr)

    return avg_loss



if __name__ == "__main__":
    # Loading Dataset
    """
    train_image_dir = "overfit_images/"
    dev_image_dir = "overfit_images/"
    train_annotations = "overfit_annotations.txt"
    dev_annotations = "overfit_annotations.txt"
    """
    train_image_dir = "train_images/"
    dev_image_dir = "dev_images/"
    train_annotations = "train_annotations.txt"
    dev_annotations = "dev_annotations.txt"

    batch_size = 2
    vocab = create_vocab(train_annotations)

    train_data_loader = LoadData(train_image_dir, train_annotations, vocab)
    train_data_iterator = DataLoader(train_data_loader, batch_size=batch_size)
    dev_data_loader = LoadData(dev_image_dir, dev_annotations, vocab)
    dev_data_iterator = DataLoader(train_data_loader, batch_size=batch_size)

    # Initiating Captioning Model
    model = CaptionNet(len(vocab))
    if torch.cuda.is_available():
        model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train model
    train(train_data_iterator, dev_data_iterator, model, criterion, optimizer)






