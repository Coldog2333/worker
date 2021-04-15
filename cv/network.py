import torch
import torch.nn as nn
import torch.utils.data

from cv.data_provider import load_glove_embedding, Flickr_Dataset
from cv.config import *

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, output_feat=False):
        super(AlexNet, self).__init__()
        self.output_feat = output_feat
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        if self.output_feat:
            return x
        else:
            return self.classifier(x)

        
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg19_bn(pretrained=False, cv_weight_file=None, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(cv_weight_file))
    return model


# def alexnet(pretrained=False, **kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = AlexNet(**kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
#         model.load_state_dict(torch.load(CV_FILE))
#     return model


class Text_Representer(nn.Module):
    def __init__(self, embedding_matrix):
        super(Text_Representer, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.vocab_num, self.embed_dim = self.embedding_matrix.shape
        self.embedding = nn.Embedding(self.vocab_num, self.embed_dim)
        self.embedding.weight = embedding_matrix

        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=256 * 6, batch_first=True)
        self.linear = nn.Linear(in_features=256 * 6, out_features=256 * 6 * 6)

    def forward(self, sentence, length):
        x = self.embedding(sentence)
        # x = torch.sum(x, dim=1)
        # x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        _, x = self.gru(x)
        x = self.linear(x[0])
        return x


class Cross_Modal_Retriever(nn.Module):
    def __init__(self, embedding_matrix, cv_weight_file=None):
        super(Cross_Modal_Retriever, self).__init__()
        self.cv_net = AlexNet(output_feat=True)
        if cv_weight_file:
            self.cv_net.load_state_dict(torch.load(cv_weight_file))
        self.nlp_net = Text_Representer(embedding_matrix=embedding_matrix)

    def forward(self, image, pos_caption, pos_length, neg_captions, neg_lengths):
        cv_feats = self.cv_net(image)
        nlp_pos_feats = self.nlp_net(pos_caption, pos_length.view(-1))
        neg_pos_feats = self.nlp_net(neg_captions.view(-1, neg_captions.shape[2]), neg_lengths.view(-1))
        neg_pos_feats = neg_pos_feats.view(neg_captions.shape[0], neg_captions.shape[1], -1)
        # similarity = torch.diagonal(torch.matmul(nlp_feats, cv_feats.T))
        return cv_feats, nlp_pos_feats, neg_pos_feats


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


if __name__ == "__main__":
    flickr_root_dir = '/Users/jiangjunfeng/Desktop/coldog/worker/dataset/flickr_30k/flickr100'
    flickr_caption_filename = flickr_root_dir + '/results.token'
    flickr_image_dir = flickr_root_dir + '/flickr100-images'

    ##### debug 词向量部分
    embedding_matrix, word2id_dict, _ = load_glove_embedding(GLOVE_FILE)
    # print(embedding_matrix.shape)
    ##### debug部分
    retriever = Cross_Modal_Retriever(embedding_matrix=nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float),
                                                                    requires_grad=True),
                                      cv_weight_file=CV_FILE)
    dataset = Flickr_Dataset(flickr_caption_filename, flickr_image_dir, word2id_dict=word2id_dict)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=2,
                                             shuffle=True,
                                             num_workers=2)

    for step, (caption, image) in enumerate(dataloader):
        sim = retriever(caption, image)
        print(sim)
