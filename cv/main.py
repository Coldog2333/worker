#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataset

from cv.network import AlexNet, Cross_Modal_Retriever, TripletLoss
from cv.data_provider import load_glove_embedding
from cv.data_provider import Flickr_Dataset_MemoryFriendly as Flickr_Dataset
from cv.config import *


if __name__ == '__main__':
    # flickr_root_dir = '/home/ubuntu/worker/dataset/flickr100'
    # flickr_caption_filename = flickr_root_dir + '/results.token'
    # flickr_image_dir = flickr_root_dir + '/flickr100-images'

    flickr_root_dir = '/home/ubuntu/worker/dataset/flickr30k'
    flickr_caption_filename = flickr_root_dir + '/results_20130124.token'
    flickr_image_dir = flickr_root_dir + '/flickr30k-images'

    ##### debug 词向量部分
    embedding_matrix, word2id_dict, _ = load_glove_embedding(GLOVE_FILE)
    # print(embedding_matrix.shape)
    ##### debug部分
    retriever = Cross_Modal_Retriever(embedding_matrix=nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float),
                                                                    requires_grad=True),
                                      cv_weight_file=CV_FILE).cuda()
    # train_dataset = Flickr_Dataset(flickr_caption_filename, flickr_image_dir, word2id_dict=word2id_dict)
    # test_dataset = Flickr_Dataset(flickr_caption_filename, flickr_image_dir, word2id_dict=word2id_dict)
    all_dataset = Flickr_Dataset(flickr_caption_filename, flickr_image_dir, word2id_dict=word2id_dict)
        
    dataset_size = len(all_dataset)
    train_size, test_size = int(0.8 * dataset_size), int(0.2 * dataset_size)
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=512,
                                             shuffle=True,
                                             num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=2)

    loss_function = TripletLoss(margin=None)
    optimizer = torch.optim.Adam(params=retriever.parameters(), lr=1e-4)

    for epoch in range(EPOCH):
        total_loss = []
        retriever.train()
        for step, (image, pos_caption, pos_length, neg_captions, neg_lengths) in enumerate(train_dataloader):
            [image, pos_caption, neg_captions] = [tensor.cuda() for tensor in [image, pos_caption,  neg_captions]]
            cv_feats, nlp_pos_feat, nlp_neg_feats = retriever(image, pos_caption, pos_length, neg_captions, neg_lengths)

            print_losses = []

            loss = loss_function(cv_feats, nlp_pos_feat, nlp_neg_feats[:, 0, :])
            # total_loss = loss
            # print_losses.append(loss.item())
            for i in range(1, K):
                loss += loss_function(cv_feats, nlp_pos_feat, nlp_neg_feats[:, i, :])
                # total_loss += loss
                # print_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            if step % 5 == 0:
                print('    step %s: loss is %s' % (step + 1, np.mean(total_loss)))

        print('EPOCH %s: Loss is %s' % ((epoch + 1), np.mean(total_loss)))

        results = []
        for step, (image, pos_caption, pos_length, neg_captions, neg_lengths) in enumerate(test_dataloader):
            [image, pos_caption, neg_captions] = [tensor.cuda() for tensor in [image, pos_caption,  neg_captions]]
            retriever.eval()
            cv_feats, nlp_pos_feat, nlp_neg_feats = retriever(image, pos_caption, pos_length, neg_captions, neg_lengths)

            dists = [torch.norm(cv_feats - nlp_pos_feat, 2, dim=1).view(-1).item()]
            for i in range(nlp_neg_feats.shape[1]):
                dist = torch.norm(cv_feats - nlp_neg_feats[:, i, :], 2, dim=1).view(-1)
                dists.append(dist.item())

            if np.argmin(dists) != 0:
                results.append(0)
            else:
                results.append(1)

        print("acc: %s" % (np.mean(results)))
