from DataIterator import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import torch
from model import *
#from attention_model import *
import time
import numpy as np
from train import create_vocab
import cv2


def text(x, vocab):
    text = ""
    for el in x:
        el = el.item()
        if el < 138:
            text += vocab[el]
    return text

if __name__ == "__main__":
    # Loading Dataset
    #image_dir = "text_images"
    #image_dir = "dev_images/"
    image_dir = "dev_images/"
    #annotations = "annotations.txt"
    #annotations = "dev_annotations.txt"
    annotations = "dev_annotations.txt"
    #annotations = "train_annotations.txt"
    batch_size = 1
    vocab = create_vocab("train_annotations.txt")
    inv_vocab = {v: k for k, v in vocab.items()}
    train_data_loader = LoadData(image_dir, annotations, vocab)
    train_data_iterator = DataLoader(train_data_loader, batch_size=batch_size)

    # Initiating Captioning Model
    model = CaptionNet(len(vocab))
    model.load_state_dict(torch.load("checkpoints/model_full94.chkpt", map_location='cpu'))
    if torch.cuda.is_available():
        model.cuda()
    t = tqdm(train_data_iterator, mininterval=1, desc='-(Validation)', leave=False)
    for batch in t:
        img_name, images, labels = batch
        pred, _ = model(images.type(torch.FloatTensor), labels)
        predictions = pred.argmax(2).contiguous()

        for im, label, pred in zip(images, labels, predictions):
            print(text(label, inv_vocab))
            print(text(pred, inv_vocab))
            cv2.imshow(img_name[0], im.data.numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        #if torch.cuda.is_available():
        #    images = Variable(images.cuda())
        #    labels = Variable(labels.cuda())
        #gold = labels.contiguous().view(-1)
        #pred, fpred = model(images, labels)
        #predictions = pred.argmax(2).contiguous().view(-1)
