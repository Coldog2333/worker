import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

from config import *

# flickr_30k_root_dir = '/Users/jiangjunfeng/Desktop/coldog/worker/dataset/flickr_30k'
# flickr_30k_caption_filename = flickr_30k_root_dir + '/flickr30k/results_20130124.token'
# flickr_30k_image_dir = flickr_30k_root_dir + '/flickr30k-images'

flickr_root_dir = '/Users/jiangjunfeng/Desktop/coldog/worker/dataset/flickr_30k/flickr100'
flickr_caption_filename = flickr_root_dir + '/results.token'
flickr_image_dir = flickr_root_dir + '/flickr100-images'

# glove_file = '/Users/jiangjunfeng/Desktop/coldog/worker/model/glove.42B.300d.txt'
glove_file = GLOVE_FILE


def load_glove_embedding(glove_file):
    # 0: pad, 1: unk
    # dim: 300
    glove_embeddings = [np.zeros(300), np.random.rand(300)]
    word2id_dict = defaultdict(int)
    id2word_dict = dict()
    with open(glove_file) as f:
        for index, line in enumerate(f):
            items = line.strip().split(' ')
            word, embedding = items[0], np.array([float(v) for v in items[1:]])
            glove_embeddings.append(embedding)
            word2id_dict[word] = index
            id2word_dict[index] = word
    return np.array(glove_embeddings), word2id_dict, id2word_dict


def load_caption(flickr_caption_filename, tokenize=False, numerize=False, word2id_dict=None):
    assert (not (tokenize == False and numerize == True))   # 想要numerize，那么必须先tokenize

    captions = defaultdict(list)
    lengths = defaultdict(list)
    with open(flickr_caption_filename) as f:
        for line in tqdm(f.readlines()):
            image_filename, caption = line.strip().split('\t')
            image_filename = image_filename[:-2]
            if tokenize:
                caption = caption.strip().split(' ')
            if numerize:
                caption = [word2id_dict[w.lower()] for w in caption]
                length = min(MAX_SEQ_LEN, len(caption))
                caption = caption[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(caption))
            captions[image_filename].append(caption)
            lengths[image_filename].append(length)
    return captions, lengths


def load_images(flickr_image_dir):
    images = dict()
    for image_filename in tqdm(os.listdir(flickr_image_dir)):
        if image_filename == 'readme.txt':
            continue
        image = Image.open(os.path.join(flickr_image_dir, image_filename))
        image = image.resize((227, 227))
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0             # 归一化到[-1, 1]之间
        images[image_filename] = image
    return images


def load_single_image(image_path):
    image = Image.open(image_path)
    image = image.resize((227, 227))
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0  # 归一化到[-1, 1]之间
    return image


class Flickr_Dataset(torch.utils.data.Dataset):
    def __init__(self, flickr_caption_filename, flickr_image_dir, word2id_dict=None):
        super(Flickr_Dataset, self).__init__()
        captions_dict, lengths_dict = load_caption(flickr_caption_filename, tokenize=True, numerize=True, word2id_dict=word2id_dict)
        images_dict = load_images(flickr_image_dir)
        filenames = images_dict.keys()

        self.captions, self.images, self.lengths = [], [], []
        for filename in filenames:
            for caption, length in zip(captions_dict[filename], lengths_dict[filename]):
                self.captions.append(caption)
                self.lengths.append(length)
                self.images.append(images_dict[filename])

        self.pos_captions = torch.tensor(self.captions, dtype=torch.long)
        self.pos_lengths = torch.tensor(self.lengths, dtype=torch.int64)
        self.images = torch.tensor(self.images, dtype=torch.float).squeeze().permute(0, 3, 1, 2)
        self.neg_captions, self.neg_lengths = [], []
        for _ in range(K):
            index = list(range(self.pos_captions.shape[0]))
            random.shuffle(index)
            self.neg_captions.append(self.pos_captions[index])
            self.neg_lengths.append(self.pos_lengths[index])
        self.neg_captions = torch.stack(self.neg_captions, dim=1)
        self.neg_lengths = torch.stack(self.neg_lengths, dim=1)

    def __getitem__(self, i):
        return self.images[i], self.pos_captions[i], self.pos_lengths[i], self.neg_captions[i], self.neg_lengths[i]

    def __len__(self):
        return len(self.captions)


class Flickr_Dataset_MemoryFriendly(torch.utils.data.Dataset):
    def __init__(self, flickr_caption_filename, flickr_image_dir, word2id_dict=None):
        captions_dict, lengths_dict = load_caption(flickr_caption_filename, tokenize=True, numerize=True, word2id_dict=word2id_dict)
        self.flickr_image_dir = flickr_image_dir
        image_filenames = os.listdir(flickr_image_dir)

        self.captions, self.images, self.lengths = [], [], []
        for filename in image_filenames:
            for caption, length in zip(captions_dict[filename], lengths_dict[filename]):
                self.captions.append(caption)
                self.lengths.append(length)
                self.images.append(os.path.join(self.flickr_image_dir, filename))

        self.pos_captions = torch.tensor(self.captions, dtype=torch.long)
        self.pos_lengths = torch.tensor(self.lengths, dtype=torch.int64)
        self.neg_captions, self.neg_lengths = [], []
        for _ in range(K):
            index = list(range(self.pos_captions.shape[0]))
            random.shuffle(index)
            self.neg_captions.append(self.pos_captions[index])
            self.neg_lengths.append(self.pos_lengths[index])
        self.neg_captions = torch.stack(self.neg_captions, dim=1)
        self.neg_lengths = torch.stack(self.neg_lengths, dim=1)

    def __getitem__(self, i):
        input_image = load_single_image(self.images[i])
        input_image = torch.tensor(input_image, dtype=torch.float).squeeze().permute(2, 0, 1)
        return input_image, self.pos_captions[i], self.pos_lengths[i], self.neg_captions[i], self.neg_lengths[i]

    def __len__(self):
        return len(self.captions)


if __name__ == '__main__':
    # captions = load_caption(flickr_caption_filename)
    # images = load_images(flickr_image_dir)
    glove_embeddings, word2id_dict, id2word_dict = load_glove_embedding(glove_file)
    dataset = Flickr_Dataset(flickr_caption_filename, flickr_image_dir, word2id_dict=word2id_dict)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=2,
                                             shuffle=True,
                                             num_workers=2)

    for step, (image, pos_caption, neg_captions) in enumerate(dataloader):
        print(image)
        print(pos_caption)
        print(neg_captions)
        print(image.shape, pos_caption.shape, neg_captions.shape)
        break
