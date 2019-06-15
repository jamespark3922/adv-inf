from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .Attention import Attention

def concat_scores(scores):
    return torch.cat([score.unsqueeze(1) for score in scores], 1)

def make_one_hot_encoding(seq,vocab_size):
    sent_onehot = torch.zeros(seq.size(0),vocab_size).cuda()
    sent_onehot.scatter_(1,seq,1)
    sent_onehot[:,0] = 0
    return sent_onehot

class NonLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(NonLinearLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_size, out_size),
            nn.Tanh()
        )

    def forward(self,emb):
        return self.main(emb)


class Classifier(nn.Module):
    def __init__(self, emb_size):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(emb_size, 1),
            nn.Sigmoid()
        )

    def forward(self,emb):
        return self.main(emb)

class HybridDiscriminator(nn.Module):
    def __init__(self,opt):
        super(HybridDiscriminator,self).__init__()
        self.opt = opt
        self.visual = None
        self.lang = LanguageModel(opt)
        self.visual = MultiModalAttEarlyFusion(opt)
        self.par = ParagraphModel(opt)

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'visual')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, 'forward_' + mode)(*args, **kwargs)

    def forward_visual(self, fc_feats, img_feats, box_feats, activity_labels, seq):
        return self.visual(fc_feats, img_feats, box_feats, activity_labels, seq)

    def forward_lang(self, seq):
        return self.lang(seq)

    def forward_par(self, seq):
        return self.par(seq)

    def use_context(self):
        self.context = True
    def get_moe_weights(self,seq):
        return self.visual.get_moe_weights(seq)

# low rank bilinear pooling
class JointEmbedVideoModel2(nn.Module):
    def __init__(self, rnn_size):
        super(JointEmbedVideoModel2, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(rnn_size, 1),
            nn.Sigmoid()
        )

    def forward(self,visual,sent):
        return self.classify(visual * sent)

class MultiModalAttEarlyFusion(nn.Module):
    def __init__(self, opt):
        super(MultiModalAttEarlyFusion, self).__init__()
        self.vocab_size = opt.vocab_size
        self.seq_length = opt.seq_length

        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.d_rnn_size
        if self.rnn_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif self.rnn_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        self.input_encoding_size = opt.d_input_encoding_size
        self.use_mean = opt.use_mean

        # video features
        self.use_video = opt.d_use_video
        self.fc_feat_size = opt.fc_feat_size
        if self.use_video:
            self.frame_embed = nn.Linear(self.fc_feat_size, self.rnn_size)
            self.video_attention = Attention(self.rnn_size)
            self.video_classifier = JointEmbedVideoModel2(self.rnn_size)

        # img features
        self.use_img = opt.d_use_img
        self.img_feat_size = opt.img_feat_size
        if self.use_img:
            self.img_embed = nn.Linear(self.img_feat_size, self.rnn_size)
            self.img_attention = Attention(self.rnn_size)
            self.img_classifier = JointEmbedVideoModel2(self.rnn_size)

        # box features
        self.use_box = opt.d_use_box
        self.box_feat_size = opt.box_feat_size
        if self.use_box:
            self.box_embed = nn.Linear(self.box_feat_size, self.rnn_size)
            self.box_attention = Attention(self.rnn_size)
            self.box_classifier = JointEmbedVideoModel2(self.rnn_size)

        # activity labels
        self.use_activity_labels = opt.d_use_activity_labels
        self.activity_size = opt.activity_size
        self.activity_encoding_size = opt.activity_encoding_size
        if self.use_activity_labels:
            self.activity_embed = nn.Linear(self.activity_size, self.rnn_size)
            self.activity_classifier = JointEmbedVideoModel2(self.rnn_size)

        # caption embedding use bag of word embedding or lstm
        self.use_bow = opt.d_use_bow
        if self.use_bow:
            self.bow_emb = nn.Linear(self.vocab_size + 2, self.rnn_size)
        else:
            self.glove = opt.glove_npy
            if self.glove is not None:
                self.input_encoding_size = 300
            self.sent_rnn = self.rnn_cell(self.input_encoding_size, self.rnn_size, self.num_layers,
                                          dropout=self.drop_prob_lm, batch_first=True)
            self.word_embed = nn.Embedding(self.vocab_size + 2, self.input_encoding_size)

        # nonlinear layer used for late fusion
        self.moe_fc = nn.Linear(self.rnn_size, self.use_video + self.use_img +
                                               self.use_box + self.use_activity_labels)

        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.use_paragraph = opt.d_use_paragraph
        self.context = False
        self.bidirectional = opt.d_bidirectional

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.use_video:
            if self.use_mean:
                self.video_encode.weight.data.uniform_(-initrange, initrange)
            else:
                self.frame_embed.weight.data.uniform_(-initrange, initrange)
        if self.use_img:
            if self.use_mean:
                self.img_encode.weight.data.uniform_(-initrange, initrange)
            else:
                self.img_embed.weight.data.uniform_(-initrange, initrange)
        if self.use_box:
            self.box_embed.weight.data.uniform_(-initrange, initrange)
        if self.use_activity_labels:
            self.activity_embed.weight.data.uniform_(-initrange, initrange)
        if not self.use_bow:
            if self.glove is not None:
                self.word_embed.load_state_dict({'weight': torch.from_numpy(np.load(self.glove))})
            else:
                self.word_embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def sent_encoder(self,seq):
        if self.use_bow:
            return self.bow_emb(make_one_hot_encoding(seq,self.vocab_size+2))
        else:
            state = self.init_hidden(seq.size(0))
            for i in range(seq.size(1)):
                it = seq[:,i].clone()
                xt = self.word_embed(it).unsqueeze(1)
                xt = self.dropout(xt)
                output, state = self.sent_rnn(xt, state)
            return self.get_hidden_state(state).squeeze(1)

    def get_hidden_state(self,state):
        if self.rnn_type == "lstm":
            return state[0].transpose(0,1)
        else:
            return state.transpose(0,1)

    def attention_encoder(self, feats, sent, mode):
        if mode == "video":
            embed = self.frame_embed
            attention = self.video_attention
        elif mode == "img":
            embed = self.img_embed
            attention = self.img_attention
        elif mode == "box":
            embed = self.box_embed
            attention = self.box_attention
        result = embed(feats)
        result = attention(sent.squeeze(1), result)
        return result

    def get_moe_weights(self,seq):
        with torch.no_grad():
            sent_embed = self.sent_encoder(seq)
            # mixture weight
            moe_weights = self.moe_fc(sent_embed)
            moe_weights = F.softmax(moe_weights, dim=1)
            return moe_weights

    def forward(self, fc_feats, img_feats, box_feats, activity_labels, seq):
        batch_size = seq.size(0)
        sent_size = seq.size(1)
        scores = []
        for n in range(sent_size):
            if seq[:,n,:].sum() == 0:
                break
            score = torch.zeros(batch_size).cuda()
            multi_score = []
            # get sent representation
            sent_embed = self.sent_encoder(seq[:,n])
            # mixture weight
            moe_weights = self.moe_fc(sent_embed)
            moe_weights = F.softmax(moe_weights, dim=1)
            if self.use_video:
                video = self.attention_encoder(fc_feats[:, n], sent_embed, 'video')
                multi_score.append(self.video_classifier(video,sent_embed))
            if self.use_img:
                image = self.attention_encoder(img_feats[:, n], sent_embed, 'img')
                multi_score.append(self.img_classifier(image,sent_embed))
            if self.use_box:
                box = self.attention_encoder(box_feats[:, n], sent_embed, 'box')
                multi_score.append(self.box_classifier(box,sent_embed))
            if self.use_activity_labels:
                if len(activity_labels.size()) == 3:
                    activity = self.activity_embed(activity_labels[:,n].float())
                else:
                    activity = self.activity_embed(activity_labels.float())
                multi_score.append(self.activity_classifier(activity, sent_embed))
            for i,m in enumerate(multi_score):
                score += (moe_weights[:, i:i + 1] * m).squeeze()
                # score += 0.33 * m.squeeze()
            scores.append(score)
        scores = concat_scores(scores)
        return scores

class LanguageModel(nn.Module):
    def __init__(self,opt):
        super(LanguageModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.seq_length = opt.seq_length

        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.d_rnn_size
        if self.rnn_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif self.rnn_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        self.input_encoding_size = opt.d_input_encoding_size

        self.glove = opt.glove_npy
        if self.glove is not None:
            self.input_encoding_size = 300
        self.word_embed = nn.Embedding(self.vocab_size + 2, self.input_encoding_size)
        self.bidirectional = opt.d_bidirectional
        self.sent_rnn = self.rnn_cell(self.input_encoding_size, self.rnn_size,
                                      self.num_layers, dropout=self.drop_prob_lm, batch_first=True,
                                      bidirectional=self.bidirectional)
        self.sent_embed = NonLinearLayer(2 * self.rnn_size, self.rnn_size, 0) if self.bidirectional \
            else NonLinearLayer(self.rnn_size, self.rnn_size, 0)
        self.classify = Classifier(self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()

    def get_hidden_state(self,state):
        if self.rnn_type == "lstm":
            return_state = state[0].transpose(0,1).contiguous()
        else:
            return_state = state.transpose(0,1).contiguous()
        if self.bidirectional:
            return_state = return_state.view(return_state.size(0),1,-1)
        return return_state

    def init_weights(self):
        initrange = 0.1
        if self.glove is not None:
            self.word_embed.load_state_dict({'weight': torch.from_numpy(np.load(self.glove))})
        else:
            self.word_embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(num_layers, bsz, self.rnn_size),
                    weight.new_zeros(num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(num_layers, bsz, self.rnn_size)

    # def pad_sentences(self, seqs):
    #     masks = torch.cat([(seqs.data.new(_seqs.size(0), 2).fill_(1).float()), (_seqs > 0).float()[:, :-1]], 1)
    #
    #     len_sents = (masks > 0).long().sum(1)
    #     len_sents, len_ix = len_sents.sort(0, descending=True)
    #
    #     inv_ix = len_ix.clone()
    #     inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(inv_ix.data)
    #
    #     new_seqs = seqs[len_ix].contiguous()
    #
    #     return new_seqs, len_sents, len_ix, inv_ix

    def forward(self, seq):
        batch_size = seq.size(0)
        sent_size = seq.size(1)
        scores = []
        for n in range(sent_size):
            if seq[:,n,:].sum() == 0:
                break
            state = self.init_hidden(batch_size)
            sent = self.dropout(self.word_embed(seq[:,n]))
            output, state = self.sent_rnn(sent, state)
            sent_embed = self.sent_embed(self.get_hidden_state(state))
            score = self.classify(sent_embed).squeeze(1).squeeze(1)
            scores.append(score)
        return concat_scores(scores)

class ParagraphModel(nn.Module):
    def __init__(self,opt):
        super(ParagraphModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.seq_length = opt.seq_length

        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.d_rnn_size
        if self.rnn_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif self.rnn_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        self.input_encoding_size = opt.d_input_encoding_size

        self.glove = opt.glove_npy
        if self.glove is not None:
            self.input_encoding_size = 300
        self.word_embed = nn.Embedding(self.vocab_size + 2, self.input_encoding_size)
        self.bidirectional = opt.d_bidirectional
        self.sent_rnn = self.rnn_cell(self.input_encoding_size, self.rnn_size,
                                      self.num_layers, dropout=self.drop_prob_lm, batch_first=True,
                                      bidirectional=self.bidirectional)
        self.classifier = Classifier(4*self.rnn_size) if self.bidirectional else Classifier(2*self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()

    def get_hidden_state(self,state):
        if self.rnn_type == "lstm":
            return_state = state[0].transpose(0,1).contiguous()
        else:
            return_state = state.transpose(0,1).contiguous()
        if self.bidirectional:
            return_state = return_state.view(return_state.size(0),1,-1)
        return return_state

    def init_weights(self):
        initrange = 0.1
        if self.glove is not None:
            self.word_embed.load_state_dict({'weight': torch.from_numpy(np.load(self.glove))})
        else:
            self.word_embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(num_layers, bsz, self.rnn_size),
                    weight.new_zeros(num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(num_layers, bsz, self.rnn_size)

    def pairwise_score(self,bsz,seq,ix1,ix2,p_scores,use):
        new_scores2 = self.classifier(torch.cat((seq[ix1],seq[ix2]),dim=2)).squeeze(1)
        for i in range(bsz):
            if use[i]:
                p_scores[i,ix2] = p_scores[i,ix2] + new_scores2[i]

    def forward(self, seq):
        batch_size = seq.size(0)
        sent_size = seq.size(1)
        sent_num = [0] * batch_size
        p_scores = seq.new_zeros(batch_size,sent_size).float()
        sents = []
        max_sent_size = sent_size

        for n in range(sent_size):
            if seq[:,n,:].sum() == 0:
                max_sent_size = n
                break

            # first embed sentence with lstm
            state = self.init_hidden(batch_size)
            sent = self.dropout(self.word_embed(seq[:, n]))
            output, state = self.sent_rnn(sent, state)

            # get number of sentences per video, so we don't consider 0-padded sentences
            sents.append(self.get_hidden_state(state))
            for i in range(batch_size):
                if seq[i,n].sum() != 0:
                    sent_num[i]+=1
        for i in range(max_sent_size-1):
            j = i+1
            # use = [(sent_num[k] > i and sent_num[k] > j) for k in range(batch_size)]
            p_score = self.classifier(torch.cat((sents[i],sents[j]),dim=2)).squeeze(1).squeeze(1)
            p_scores[:, i + 1] = p_score
        return p_scores