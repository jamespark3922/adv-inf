from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random


def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq,add_punct=False):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if add_punct: # for bert
            txt = txt + '.'
        txt = unicode(txt.encode('ascii', 'ignore'))
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

# split to batch_size * sent_num  x rest
def align_seq(sent_num,seq):
    # seq      = batch_size x sent_num x rest
    # aligned  = total_sent x rest
    batch_size = seq.size(0)
    total_sent = sum(sent_num)
    if len(seq.size()) >= 3:
        new_shape = [total_sent] + list(seq.size()[2:])
    else:
        new_shape = total_sent
    seq_aligned = seq.new_zeros(new_shape)

    cur = 0
    for i in range(batch_size):
        i_n = min(sent_num[i],seq[i].size(0))
        seq_aligned[cur: cur + i_n] = seq[i,:i_n]
        cur+=sent_num[i]

    return seq_aligned.contiguous()

# combine to batch_size x rest
def combine_seq(sent_num,seq):
    # seq      = batch_size x sent_num x seq_length
    # combined = batch_size x (sent_num * seq_length)
    assert len(seq.size()) == 3
    batch_size = seq.size(0)
    seq_combined = seq.new_zeros(batch_size, max(sent_num) * seq.size(2))
    cur = 0
    for i in range(batch_size):
        combined = seq[i,:sent_num[i]].view(-1)
        seq_combined[i,:combined.size(0)] = combined
    return seq_combined

def generate_paragraph_mask(sent_num,seq):
    assert len(seq.size()) == 3
    batch_size = seq.size(0)
    mask = seq.new_zeros(seq.size())
    ones = torch.ones(seq.size(2)).cuda()
    for i in range(batch_size):
        mask[i,:sent_num[i]] = ones.expand(sent_num[i],-1)
    return mask

def get_bert_masks(sent_num,seq,tokenizer,vocab,use_pair=False,eval=False,max_seq_length=40+2):
    input_ids = []
    input_masks = []
    input_tokens = []
    seq = align_seq(sent_num, seq)
    txts = decode_sequence(vocab, seq, add_punct=True)

    if not use_pair:
        for txt in txts:
            tokens_a = tokenizer.tokenize(txt)
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            input_token = [0] * len(input_id)
            padding = [0] * (max_seq_length - len(input_id))
            input_id += padding
            input_mask += padding
            input_token += padding

            assert len(input_id) == max_seq_length, \
                "input length: %d does not match maximum sequence length: %d" % (len(input_id), max_seq_length)
            assert len(input_mask) == max_seq_length, \
                "mask length: %d does not match maximum sequence length: %d" % (len(input_mask), max_seq_length)

            input_ids.append(input_id)
            input_masks.append(input_mask)
            input_tokens.append(input_token)
    else:
        prev_txt = ["[PAD]"]
        total_max_length = max_seq_length * 2 - 1
        sent_cnt = 0
        sent_idx = 0
        for txt in txts:
            tokens_a = tokenizer.tokenize(txt)
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            tokens = ["[CLS]"] + prev_txt + ["[SEP]"] + tokens_a + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            input_token = [0] * (len(prev_txt) + 2) + [1] * (len(tokens_a) + 1)
            prev_txt = tokens_a

            if eval or sent_cnt > 0: # only add if it's not the first sentence
                padding = [0] * (total_max_length - len(input_id))
                input_id += padding
                input_mask += padding
                input_token += padding

                assert len(input_id) == total_max_length, \
                "input length: %d does not match maximum sequence length: %d" % (len(input_id), total_max_length)
                assert len(input_mask) == total_max_length, \
                "mask length: %d does not match maximum sequence length: %d" % (len(input_mask), total_max_length)

                input_ids.append(input_id)
                input_masks.append(input_mask)
                input_tokens.append(input_token)

            sent_cnt+=1
            if sent_cnt == sent_num[sent_idx]:
                sent_cnt = 0
                sent_idx+=1
                prev_txt = ["[PAD]"]

    input_ids = torch.tensor(input_ids, dtype=torch.long).cuda().contiguous()
    input_masks = torch.tensor(input_masks, dtype=torch.long).cuda().contiguous()
    input_tokens = torch.tensor(input_tokens, dtype=torch.long).cuda().contiguous()
    return input_ids, input_masks, input_tokens

def get_neg_lang(sent_num,gt,gen): # choose gt or gen

    # replace part of sentence with previous phrase.
    def repeat_words(seq, first_zero):
        l = np.random.randint(3, 6) # length of phrase to repeat
        source = np.random.choice(range(first_zero - 2 * l)) # source index to copy from
        r1 = np.random.randint(3, 5)  # offset between source and target
        target = min(source + l + r1, first_zero - l) # target index to start copying
        seq[target:target + l] = seq[source:source + l].copy()
        return seq

    new_seq = np.zeros(gen.size()).astype(long)
    batch_size = new_seq.shape[0]
    for i in range(batch_size):
        r = np.random.randint(0, 2)
        for j in range(sent_num[i]):

            # choose gt or gen to modify
            if i % 3 == 0:
                seq = gen[i,j].clone()
            else:
                seq = gt[i,j,1:-1].clone()
            seq = seq.contiguous().data.cpu().numpy()

            first_zero = np.argmax(seq == 0)
            if first_zero == 0:
                first_zero = new_seq.shape[2]

            # insert repeating phrases
            if first_zero > 12 and r % 3 == 0 and first_zero < 24:
                new_seq[i,j] = repeat_words(seq,first_zero)

            # swap few words within a sentence
            else:
                if first_zero > 8:
                    to_swap = sorted(np.random.choice(first_zero, i % 3 + 3,replace=False))
                    while True:
                        swapped = np.random.permutation(to_swap)
                        if not np.array_equal(swapped, to_swap):
                            break
                    seq[to_swap] = seq[swapped]
                    new_seq[i,j] = seq

                else:
                    shuffle = seq[:first_zero]
                    while True:
                        pw = np.random.permutation(shuffle)
                        if len(pw) <= 1 or (not np.array_equal(pw,new_seq[i,j,:first_zero])):
                            new_seq[i,j,:first_zero] = pw
                            break
    return torch.from_numpy(new_seq).cuda()

# https://stackoverflow.com/questions/25200220/generate-a-random-derangement-of-a-list
def random_derangement(n):
    if n == 0:
        return 0
    while True:
        v = range(n)
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return v

def get_neg_pair(sent_num,seq):
    new_seq = seq.copy()
    batch_size = seq.shape[0]
    for i in range(batch_size):
        r = np.random.choice(batch_size, 1)[0]

        # only consider more than two sentences
        if sent_num[i] > 1:

            # repeat two sentences
            if i % 2 == 0 and sent_num[i] == 2:
                if i % 2 == 1:
                    new_seq[i,1] = new_seq[i,0]
                else:
                    new_seq[i,0] = new_seq[i,1]

            # derangement
            elif r % 3 == 0:
                deranged = random_derangement(sent_num[i])
                new_seq[i,:sent_num[i]] = new_seq[i][deranged]

            # mismatched sentence within a batch
            else:
                for j in range(sent_num[i]):
                    tochoose = range(batch_size)
                    tochoose.remove(i)
                    r1 = np.random.choice(tochoose,1)[0]
                    r2 = np.random.choice(sent_num[r1],1)[0]
                    first_zero = np.argmax(new_seq[i,j-1,1:] == 0)

                    # repeating sentence with cut off phrase
                    if j > 0 and r2 % 3 == 0 and first_zero > 8:
                        new_seq[i,j] = new_seq[i,j-1].copy()
                        new_seq[i,j,8-(r2 % 3):] = 0

                    # mismatched pair
                    else:
                        new_seq[i,j] = seq[r1,r2].copy()
    return new_seq

def dense_classifier(sent_num,fc_feats,img_feats,classifier):
    new_sent_num = [max(sent_num)] * len(sent_num)
    new_fc_feats = align_seq(new_sent_num, fc_feats)
    new_img_feats = align_seq(new_sent_num, img_feats)
    activities = classifier(new_fc_feats,new_img_feats)
    activities = activities.view(fc_feats.size(0),max(sent_num),-1)
    return activities

def load_dis_model_param(model,dict,modules=['visual','lang','par']):
    for k,v in dict.items():
        if k.split('.')[0] not in modules:
            print(k)
            dict[k] = model.state_dict()[k] # use initialized value instead
    model.load_state_dict(dict)

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
