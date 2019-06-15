from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel
from .Attention import Attention
import numpy as np
import time


class MultiModalGenerator(CaptionModel):
    def __init__(self, opt):
        super(MultiModalGenerator, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.video_encoding_size = opt.video_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        if self.rnn_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif self.rnn_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.use_mean = opt.use_mean

        # motion features
        self.use_video = opt.use_video
        self.fc_feat_size = opt.fc_feat_size
        if self.use_video:
            self.frame_embed = nn.Linear(self.fc_feat_size, self.rnn_size)
            self.video_attention = Attention(self.rnn_size)

        # img features
        self.use_img = opt.use_img
        self.img_feat_size = opt.img_feat_size
        if self.use_img:
            self.image_embed = nn.Linear(self.img_feat_size, self.rnn_size)
            self.img_attention = Attention(self.rnn_size)

        # box features
        self.use_box = opt.use_box
        self.box_feat_size = opt.box_feat_size
        self.box_seg = opt.box_seg
        if self.use_box:
            self.box_embed = nn.Linear(self.box_feat_size, self.rnn_size)
            self.box_attention = Attention(self.rnn_size)

        # activity labels
        self.use_activity_labels = opt.use_activity_labels # False by default
        self.activity_size = opt.activity_size
        self.activity_encoding_size = opt.activity_encoding_size
        if self.use_activity_labels:
            self.activity_embed = nn.Linear(self.activity_size, self.activity_encoding_size)

        # sent embed
        self.glove = None # opt.glove_npy
        if self.glove is not None:
            self.input_encoding_size = 300
        self.word_embed = nn.Embedding(self.vocab_size + 2, self.input_encoding_size)
        self.context_encoding_size = self.rnn_size
        self.sent_rnn = self.rnn_cell(2 * self.input_encoding_size + self.rnn_size, self.rnn_size,
                                      self.num_layers, dropout=self.drop_prob_lm, batch_first=True)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 2)

        # context
        self.context = False
        self.context_embed = None

        # entire encoder
        self.encoder = nn.Linear(self.activity_encoding_size + 3 * self.rnn_size, self.input_encoding_size)

        self.dropout = nn.Dropout(self.drop_prob_lm)
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
                self.image_encode.weight.data.uniform_(-initrange, initrange)
            else:
                self.image_embed.weight.data.uniform_(-initrange, initrange)
        if self.use_box:
            self.box_embed.weight.data.uniform_(-initrange, initrange)
        if self.use_activity_labels:
            self.activity_embed.weight.data.uniform_(-initrange, initrange)
        if self.glove is not None:
            self.word_embed.load_state_dict({'weight': torch.from_numpy(np.load(self.glove))})

        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def get_hidden_state(self,state):
        if self.rnn_type == "lstm":
            return state[0].transpose(0,1).cuda()
        else:
            return state.transpose(0,1).cuda()

    # def attention_encoder(self, feats, state, embed, attention):
    #     result = embed(feats)
    #     result = attention(self.get_hidden_state(state).squeeze(1), result).unsqueeze(1)
    #     return result

    def attention_encoder(self, feats, state, mode):
        if mode == "video":
            result = self.frame_embed(feats)
            attention = self.video_attention
        elif mode == "img":
            result = self.image_embed(feats)
            attention = self.img_attention
        elif mode == "box":
            if self.use_box == 2:
                feats = feats.view(feats.size(0),3,-1,feats.size(-1))
                result = torch.max(self.box_embed(feats),dim=2)[0]
            else:
                result = self.box_embed(feats)

            attention = self.box_attention
        result = attention(self.get_hidden_state(state).squeeze(1), result).unsqueeze(1)
        return result

    def use_context(self):
        self.context = True

    def _forward(self, fc_feats, img_feats, box_feats, activity_labels, seq):
        # fc_feats = batch_size x sent_num x frame_num x feat_dim
        # seq = batch_size x sent_num x seq_length

        batch_size = fc_feats.size(0)
        sent_num = fc_feats.size(1)
        outputs = []
        video = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        image = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        box = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        activity = torch.zeros(batch_size,1,self.activity_encoding_size).cuda()
        context = torch.zeros(batch_size,1,self.context_encoding_size).cuda()
        for n in range(sent_num):
            if fc_feats[:,n,:,:].sum() == 0:
                break
            # decoder initalization
            sequence = []
            state = self.init_hidden(batch_size)
            if self.use_activity_labels: # False by default to avoid cheating
                if len(activity_labels.size()) == 3:
                    activity = self.activity_embed(activity_labels[:,n].float()).unsqueeze(1)
                else:
                    activity = self.activity_embed(activity_labels.float()).unsqueeze(1)
            for i in range(seq.size(2)-1):
                it = seq[:,n,i].clone()
                # break if all the sequences end
                if i >= 1 and seq[:,:,i].sum() == 0:
                    break
                if self.use_video:
                    video = self.attention_encoder(fc_feats[:, n], state, 'video')
                if self.use_img:
                    image = self.attention_encoder(img_feats[:, n], state, 'img')
                if self.use_box:
                    box = self.attention_encoder(box_feats[:, n], state, 'box')
                encoded = self.encoder(torch.cat((video, image, box, activity), dim=2))
                xt = self.word_embed(it).unsqueeze(1)
                xt = torch.cat((encoded,context,xt),dim=2)
                output, state = self.sent_rnn(xt, state)
                output = F.log_softmax(self.logit(self.dropout(output.squeeze(1))), dim=1)
                sequence.append(output)
            if self.context:
                context = self.get_hidden_state(state)
            outputs.append(torch.cat([_.unsqueeze(1) for _ in sequence], 1).contiguous())
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

    def _sample(self, fc_feats, img_feats, box_feats, activity_labels, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if sample_max and beam_size > 1:
            return self._sample_beam(fc_feats, img_feats, box_feats, activity_labels, opt=opt)
        batch_size = fc_feats.size(0)
        sent_num = fc_feats.size(1)
        video = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        image = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        box = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        context = torch.zeros(batch_size,1,self.rnn_size).cuda()
        activity = torch.zeros(batch_size,1,self.activity_encoding_size).cuda()
        seq = fc_feats.new_zeros(batch_size,sent_num, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size,sent_num, self.seq_length)
        for n in range(sent_num):
            if fc_feats[:,n,:,:].sum() == 0:
                break
            state = self.init_hidden(batch_size)
            if self.use_activity_labels:
                if len(activity_labels.size()) == 3:
                    activity = self.activity_embed(activity_labels[:,n].float()).unsqueeze(1)
                else:
                    activity = self.activity_embed(activity_labels.float()).unsqueeze(1)
            for t in range(self.seq_length + 1):
                if t == 0 : # input <bos>
                    it = fc_feats.new_zeros(batch_size, dtype=torch.long)
                if self.use_video:
                    video = self.attention_encoder(fc_feats[:, n], state, 'video')
                if self.use_img:
                    image = self.attention_encoder(img_feats[:, n], state, 'img')
                if self.use_box:
                    box = self.attention_encoder(box_feats[:, n], state, 'box')
                encoded = self.encoder(torch.cat((video, image, box, activity), dim=2))
                xt = self.word_embed(it).unsqueeze(1)
                xt = torch.cat((encoded,context,xt),dim=2)
                output, state = self.sent_rnn(xt, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))), dim=1)

                # sample the next_word
                if t == self.seq_length: # skip if we achieve maximum length
                    break
                if sample_max:
                    sampleLogprobs_2, it_2 = torch.topk(logprobs.data, 2, dim=1)
                    it = it_2.new_zeros(batch_size)
                    sampleLogprobs = it_2.new_zeros(batch_size)
                    fc_feats.new_zeros(batch_size)

                    # deal with <unk>
                    for i,b in enumerate(it_2):
                        it[i] = b[0]
                        sampleLogprobs[i] = sampleLogprobs_2[i,0]
                        if b[0].item() == (self.vocab_size+1):
                            it[i] = b[1]
                            sampleLogprobs[i] = sampleLogprobs_2[i,1]
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                    it = it.view(-1).long()  # and flatten indices for downstream processing
                # stop when all finished
                if t == 0:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)

                it = it * unfinished.type_as(it)
                seq[:,n,t] = it  # seq[t] the input of t+2 time step
                seqLogprobs[:,n,t] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break
            if self.context:
                context = self.get_hidden_state(state)
        return seq,seqLogprobs

    # Not implemented yet
    def _sample_beam(self, fc_feats, img_feats, box_feats, activity_labels, opt={}):
        beam_size = opt.get('beam_size', 5)
        batch_size = fc_feats.size(0)
        sent_num = fc_feats.size(1)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(batch_size, sent_num, self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, sent_num, self.seq_length)
        video = torch.zeros(beam_size, 1, self.rnn_size).cuda()
        image = torch.zeros(beam_size, 1, self.rnn_size).cuda()
        box = torch.zeros(beam_size, 1, self.rnn_size).cuda()
        activity = torch.zeros(beam_size, 1, self.activity_encoding_size).cuda()
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            context = torch.zeros(beam_size, 1, self.rnn_size).cuda()
            for n in range(sent_num):  # sent_num
                if fc_feats[k, n, :, :].sum() == 0:
                    break
                state = self.init_hidden(beam_size)
                fkn = fc_feats[k, n].expand(beam_size, -1, -1)
                ikn = img_feats[k, n].expand(beam_size, -1, -1)
                bkn = box_feats[k, n].expand(beam_size, -1, -1)
                if self.use_video:
                    video = self.attention_encoder(fkn, state, 'video')
                if self.use_img:
                    image = self.attention_encoder(ikn, state, 'img')
                if self.use_box:
                    box = self.attention_encoder(bkn, state, 'box')
                if self.use_activity_labels:
                    if len(activity_labels.size()) == 3:
                        activity = self.activity_embed(activity_labels[:, n].float()).unsqueeze(1)
                    else:
                        activity = self.activity_embed(activity_labels.float()).unsqueeze(1)
                encoded = self.encoder(torch.cat((video, image, box, activity), dim=2))

                # beam search
                it = fc_feats.new_zeros(beam_size, dtype=torch.long)
                xt = self.word_embed(it).unsqueeze(1)
                xt = torch.cat((encoded, context, xt), dim=2)
                output, state = self.sent_rnn(xt, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))), dim=1)
                self.done_beams[k] = self.beam_search(state, logprobs, fkn,ikn,bkn,activity,context, opt=opt)  # actual beam search
                seq[k, n, :] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, n, :] = self.done_beams[k][0]['logps']
                if self.context:
                    context = self.done_beams[k][0]['state'].expand(beam_size,-1,-1)
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def get_logprobs_state(self, it, fkn, ikn, bkn, activity, context, state):
        batch_size = it.size(0)
        video = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        image = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        box = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        # 'it' contains a word index
        if self.use_video:
            video = self.attention_encoder(fkn, state, 'video')
        if self.use_img:
            image = self.attention_encoder(ikn, state, 'img')
        if self.use_box:
            box = self.attention_encoder(bkn, state, 'box')
        encoded = self.encoder(torch.cat((video, image, box, activity), dim=2))
        xt = self.word_embed(it).unsqueeze(1)
        xt = torch.cat((encoded, context, xt), dim=2)
        output, state = self.sent_rnn(xt, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))), dim=1)
        return logprobs, state

    def sample_sequential(self, fc_feats, img_feats, box_feats, activity_labels, context=None, opt={}):
        temperature = opt.get('temperature', 1.0)

        batch_size = fc_feats.size(0)
        video = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        image = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        box = torch.zeros(batch_size, 1, self.rnn_size).cuda()
        activity = torch.zeros(batch_size,1,self.activity_encoding_size).cuda()
        seq = fc_feats.new_zeros(batch_size,self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size,self.seq_length)
        state = self.init_hidden(batch_size)
        if self.use_activity_labels:
            activity = self.activity_embed(activity_labels.float()).unsqueeze(1)
        if context is None:
            context = torch.zeros(batch_size,1,self.rnn_size).cuda()
        for t in range(self.seq_length + 1):
            if t == 0 : # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            if self.use_video:
                video = self.attention_encoder(fc_feats, state, 'video')
            if self.use_img:
                image = self.attention_encoder(img_feats, state, 'img')
            if self.use_box:
                box = self.attention_encoder(box_feats, state, 'box')
            encoded = self.encoder(torch.cat((video, image, box, activity), dim=2))
            xt = self.word_embed(it).unsqueeze(1)
            xt = torch.cat((encoded,context,xt),dim=2)
            output, state = self.sent_rnn(xt, state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))), dim=1)

            # sample the next_word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if temperature == 1.0:
                prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
            else:
                # scale logprobs by temperature
                prob_prev = torch.exp(torch.div(logprobs.data, temperature))

            # deal with <unk> which is at self.vocab_size + 1
            it_2 = torch.multinomial(prob_prev, 2)
            it = it_2[:,0].clone().unsqueeze(1)
            bool = torch.eq(it,it.new_ones(batch_size,1)*(self.vocab_size+1)).long()
            it = it_2.gather(1,bool)

            sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
            it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)

            seq[:,t] = it  # seq[t] the input of t+2 time step
            seqLogprobs[:,t] = sampleLogprobs.view(-1)

            if unfinished.sum() == 0:
                break

        context = self.get_hidden_state(state)
        return seq,seqLogprobs,context