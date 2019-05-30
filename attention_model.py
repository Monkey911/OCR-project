from math import sqrt

import torch.nn as nn
import torch
import pretrainedmodels
from torch.autograd import Variable



class ImageEncoder(nn.Module):

    def __init__(self, model):
        super(ImageEncoder, self).__init__()
        #print(model)
        partial_alexnet = nn.Sequential(*list(model.children())[1:-7])  # delete the last fc layers.
        self.partial_alexnet = partial_alexnet

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dense = nn.Linear(in_features=196, out_features=200)

    def forward(self, images):
        #print("encoder")
        images.transpose_(2, 3)
        images.transpose_(1, 2)
        r = self.partial_alexnet(images)
        r = r.view(r.size(0), r.size(1), -1)
        return r


        r = self.conv1(images)
        #print(r.shape)
        r = self.relu(r)
        #print(r.shape)
        r = self.maxpool(r)
        #print(r.shape)
        r = self.conv2(r)
        #print(r.shape)
        r = self.relu(r)
        #print(r.shape)
        r = self.maxpool(r)
        #print(r.shape)
        #print(r.shape)
        r = self.conv3(r)
        r = self.relu(r)
        r = self.conv4(r)
        r = self.relu(r)
        r = self.conv5(r)
        r = self.relu(r)
        r = self.maxpool(r)

        #print(r.shape)
        r = r.view(r.size(0), r.size(1), -1)
        #r = r.view(r.size(0), -1)

        #r = r.unsqueeze(2)#.repeat(1, 28)
        #print(r.shape)
        #r = r.view(r.size(0), r.size(1), -1)
        #r = self.dense(r)
        #r = r.view(r.size(0), 50, -1)
        #print(r.shape)
        #r = r.view(r.size(0), r.size(1), -1)
        return r



class Attention(nn.Module):
    def __init__(self, hs):
        super(Attention, self).__init__()

        self.w_t = nn.Linear(in_features=hs, out_features=hs)
        self.w_v = nn.Linear(in_features=hs, out_features=hs)

        self.hs = hs

        self.softmax = nn.Softmax(dim=2)

    def forward(self, t, v):

        residual = t
        t = self.w_t(t)
        v = self.w_v(v)

        attn = self.softmax(torch.bmm(t, v.transpose(1, 2)) / sqrt(float(self.hs)))
        output = torch.bmm(attn, v) + residual
        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()

        self.RNN = nn.GRU(input_size=196, hidden_size=98, num_layers=1, batch_first=False, bidirectional=True)
        self.attention = Attention(196)
        self.dense = nn.Linear(in_features=196, out_features=100)

    def forward(self, inp):
        # RNN layer
        r, _ = self.RNN(inp)
        r = self.attention(r, inp)
        r = self.dense(r)
        r = r.view(r.size(0), -1)
        return r


class CaptionNet(nn.Module):
    def __init__(self, vocab_size):
        super(CaptionNet, self).__init__()

        cnn_model = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained='imagenet')
        if torch.cuda.is_available():
            cnn_model.cuda()

        #self.text_encoder = TextEncoder(vocab_size)
        self.encoder = ImageEncoder(cnn_model)
        self.decoder = Decoder(vocab_size)
        #self.dense = nn.Linear(3456, 100)
        self.dense = nn.Linear(25600, 10000)
        self.output = nn.Linear(200, vocab_size)

    def forward(self, images, labels):
        # Encoding Images
        #t_enc = self.text_encoder(ger_captions)
        images = images.type(torch.FloatTensor)
        if torch.cuda.is_available():
            images = Variable(images.cuda())
        output = self.encoder(images)
        output = self.decoder(output)
        #print(output.shape)
        output = self.dense(output)
        output = output.view(output.size(0), 50, -1)

        output = self.output(output)
        foutput = output.contiguous().view(-1, output.size(2))
        return output, foutput

    def generate(self, images, captions, ger_captions):

        # Encoding Images
        V, g = self.encoder(images)

        # Encode ger captions
        t_enc = self.text_encoder(ger_captions)

        if torch.cuda.is_available():
            captions = captions.cuda()

        # Decoding Captions
        output = self.decoder(V, g, captions, t_enc)

        # Output Layer to project from hidden size to vocabulary size for prediction
        output = self.output(output)

        pred = output.max(2)[1]

        return pred

