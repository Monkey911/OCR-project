from torch.utils.data.dataset import Dataset
import pretrainedmodels.utils as utils

import os
import torch
import codecs
import cv2


class LoadData(Dataset):

    def __init__(self, image_dir, annotations, vocab):
        self.load_img = utils.LoadImage()
        self.image_dir = image_dir
        self.annotations = codecs.open(annotations, encoding="utf-8").readlines()
        self.vocab = vocab
        self.vocab_keys = vocab.keys()
        self.num_annotations = sum(1 for line in open(annotations))
        self.max_len = 50

    def __getitem__(self, index):
        img_name, label = self.annotations[index].split("\t")
        #img = self.load_img("text_images/" + img_path)
        img = cv2.imread(self.image_dir + img_name)
        img = self.rehsape_img(img)

        label = label.strip()
        label = [self.vocab[token] if token in self.vocab_keys else self.vocab['<UNK>'] for token in label]
        label = [self.vocab['<BOS>']] + label + [self.vocab['<EOS>']]
        label = label[:self.max_len] + (self.max_len - len(label))*[self.vocab['<PAD>']]
        label = torch.LongTensor(label)
        return img_name, img, label

    #https://gist.github.com/jdhao/f8422980355301ba30b6774f610484f2
    def rehsape_img(self, img):
        desired_size = 500
        #cv2.imshow("aa", img)
        #cv2.waitKey(0)

        old_size = img.shape[:2] # old_size is in (height, width) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        img = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        #cv2.imshow("aa", new_img)
        #cv2.waitKey(0)
        return new_img


    def __len__(self):
        return self.num_annotations
