import xml.etree.ElementTree as ET
import cv2
from os import listdir
from tqdm import tqdm

import os

#TODO change them
train_annotations_file = "train_annotations.txt"
train_images_dir = "train_images/"
dev_annotations_file = "dev_annotations.txt"
dev_images_dir = "dev_images/"

# splits the priginal dataset into many images and annotations
#file = open("train_annotations.txt", "w")
train_file = open(train_annotations_file, "w")
dev_file = open(dev_annotations_file, "w")

#doc = minidom.parse('annotations/img_11267413.xml')

counter = 0
#for f in tqdm(['img_11267413.jpg']):
for f in tqdm(listdir("images/")):
    filename, format = f.split(".")
    root = ET.parse('annotations/' + filename + '.xml').getroot()
    object_order = 0
    for labelxml in root.findall('object'):
        image_name = filename + '_' + str(object_order) + '.' + format
        object_order += 1

        label = labelxml.find("text").text
        label = label.replace("<br/>", "⌑")
        if len(label) <= 48: # 49 and 59 are start and end symbols
            if counter % 15 == 0:
                dev_file.write(image_name + "\t" + label + "\n")
            else:
                train_file.write(image_name + "\t" + label + "\n")

            coords = labelxml.findall('polygon/pt')
            x_start = int(coords[0][0].text)
            x_end = int(coords[1][0].text)
            y_start = int(coords[0][1].text)
            y_end = int(coords[2][1].text)

            img = cv2.imread('images/' + f)
            img = img[y_start:y_end, x_start:x_end]
            #cv2.imshow('image', img)
            #cv2.waitKey(0)
            if counter % 15 == 0:
                cv2.imwrite(dev_images_dir + image_name, img)
            else:
                cv2.imwrite(train_images_dir + image_name, img)
            counter += 1

train_file.close()
dev_file.close()


# removing annotations, where images are incorrect
with open(train_annotations_file, "r") as f:
    lines = f.readlines()
with open(train_annotations_file, "w") as f:
    for line in lines:
        img_path, label = line.split("\t")
        img = cv2.imread(train_images_dir + img_path)
        try:
            a = img.shape
            f.write(line)
        except:
            pass


with open(dev_annotations_file, "r") as f:
    lines = f.readlines()
with open(dev_annotations_file, "w") as f:
    for line in lines:
        img_path, label = line.split("\t")
        img = cv2.imread(train_images_dir + img_path)
        try:
            a = img.shape
            f.write(line)
        except:
            pass
