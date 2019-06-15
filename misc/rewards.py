from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor


CiderD_scorer = None
Bleu_scorer = None
Meteor_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Meteor_scorer
    Meteor_scorer = Meteor_scorer or Meteor()

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def score_sent(gen,gt,mode='cider'):
    batch_size = len(gen)
    # Prep for NLP Metrics
    res = OrderedDict()
    gts = OrderedDict()
    scores = np.zeros(batch_size)

    for i in range(batch_size):
        res[i] = [array_to_str(gen[i])]
        gts[i] = [array_to_str(gt[i])]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)] # for cider
    gts = {i: gts[i % batch_size] for i in range(batch_size)}

    # calculate language scores
    if mode == 'meteor' or mode =='hybrid':
        _, meteor_scores = Meteor_scorer.compute_score(gts, res)
        scores+=meteor_scores
    if mode != 'meteor' or mode == 'hybrid':
        _, cider_scores= CiderD_scorer.compute_score(gts, res_)
        scores+=cider_scores
    return scores

def expand_reward(sent_num,reward):
    # seq = batch_size x sent_num x seq_length
    batch_size = reward.shape[0]
    length = reward.shape[1]
    combined = np.tile(reward,(batch_size,max(sent_num) * length))
    return combined

# # for image captioning
# def get_self_critical_reward(G, D, fc_feats, att_feats, att_masks, data, gen_result, opt,print_score=False):
#     batch_size = gen_result.size(0)
#     seq_per_img = batch_size // len(data['gts'])
#
#     # get greedy decoding baseline
#     G.eval()
#     with torch.no_grad():
#         greedy_result, _ = G(fc_feats, att_feats, att_masks, mode='sample', )
#     G.train()
#
#     # Discriminator Score
#     scores = np.zeros(batch_size)
#     if opt.gan and opt.gan_reward_weight > 0:
#         with torch.no_grad():
#             D.eval()
#             gen_d = (D(fc_feats,att_feats,gen_result, att_masks))
#             greedy_d = (D(fc_feats, att_feats, greedy_result, att_masks))
#             gt_d = D(fc_feats,att_feats,torch.from_numpy(data['labels']).cuda()[:,1:-1],att_masks)
#             dis_scores = opt.gan_reward_weight * (gen_d - greedy_d)
#             scores += dis_scores.data.cpu().numpy()
#             if print_score:
#                 print('dis_gen_score', gen_d.mean().item(),
#                       'dis_greedy_score', greedy_d.mean().item(),
#                       'dis_gt_score', gt_d.mean().item())
#
#     res = OrderedDict()
#
#     gen_result_data = gen_result.data.cpu().numpy()
#     greedy_result_data = greedy_result.data.cpu().numpy()
#
#     for i in range(batch_size):
#         res[i] = [array_to_str(gen_result_data[i])]
#     for i in range(batch_size):
#         res[batch_size + i] = [array_to_str(greedy_result_data[i])]
#
#     gts = OrderedDict()
#     for i in range(len(data['gts'])):
#         gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
#
#     res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)] # for cider
#     res__ = {i: res[i] for i in range(2 * batch_size)} # for bleu
#     gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
#
#     if opt.cider_reward_weight > 0:
#         _, cider_scores = CiderD_scorer.compute_score(gts, res_)
#         print('cider_gen_score',np.mean(cider_scores[:batch_size]), 'cider_greedy_score', np.mean(cider_scores[batch_size:]))
#
#         cider_scores = opt.cider_reward_weight * cider_scores
#         cider_scores = cider_scores[:batch_size] - cider_scores[batch_size:]
#         scores+= cider_scores
#
#     if opt.meteor_reward_weight > 0:
#         _, meteor_scores = Meteor_scorer.compute_score(gts, res_)
#         #print('gen',np.mean(meteor_scores[:batch_size]), 'greedy', np.mean(meteor_scores[batch_size:]))
#
#         meteor_scores = opt.meteor_reward_weight * meteor_scores
#         meteor_scores = meteor_scores[:batch_size] - meteor_scores[batch_size:]
#         scores+= meteor_scores
#
#     rewards = np.repeat(scores[:, np.newaxis], gen_result_data.shape[1], 1)
#     return rewards

def get_self_critical_reward_video(G, D, data, gen_result, opt, mode='visual', use_ngram=False):
    # load data
    tmp = [data['fc_feats'], data['att_feats'], data['img_feats'], data['box_feats'],
           data['labels'], data['masks'], data['att_masks'], data['activities']]
    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    fc_feats, att_feats, img_feats, box_feats, labels, masks, att_masks, activities = tmp
    sent_num = data['sent_num']

    batch_size = sum(sent_num)

    # get greedy decoding baseline
    G.eval()
    with torch.no_grad():
        greedy_result, _ = G(fc_feats, att_feats, att_masks, img_feats, box_feats, activities, sent_num=sent_num,mode='sample')
    G.train()

    # Discriminator Score
    scores = np.zeros(batch_size)
    if opt.gan and opt.gan_reward_weight > 0:
        with torch.no_grad():
            D.eval()
            gen_d = (D(fc_feats, img_feats, box_feats, activities,gen_result, mode=mode))
            greedy_d = (D(fc_feats, img_feats, box_feats, activities,greedy_result, mode=mode))
            gt_d = D(fc_feats,img_feats, box_feats, activities, torch.from_numpy(data['labels']).cuda()[:,:,1:-1],mode=mode)
            dis_scores = opt.gan_reward_weight * (gen_d - greedy_d)
            scores += utils.align_seq(sent_num,dis_scores).data.cpu().numpy()
            print(mode,'dis_gen_score', utils.align_seq(sent_num,gen_d).mean().item(),
                  'dis_greedy_score', utils.align_seq(sent_num,greedy_d).mean().item(),
                  'dis_gt_score', utils.align_seq(sent_num,gt_d).mean().item())

    # Prep for NLP Metrics
    res = OrderedDict()

    with torch.no_grad():
        gen_result_data = utils.align_seq(sent_num,gen_result).data.cpu().numpy()
        greedy_result_data = utils.align_seq(sent_num,greedy_result).data.cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result_data[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_result_data[i])]

    gts = OrderedDict()
    counter = 0
    for i in range(len(data['gts'])):
        for j in range(sent_num[i]):
            gts[counter] = [array_to_str(data['gts'][i][j])]
            counter+=1

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)] # for cider
    res__ = {i: res[i] for i in range(2 * batch_size)} # for bleu
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    # calculate language scores
    if use_ngram and opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('cider_gen_score',np.mean(cider_scores[:batch_size]), 'cider_greedy_score', np.mean(cider_scores[batch_size:]))

        cider_scores = opt.cider_reward_weight * cider_scores
        cider_scores = cider_scores[:batch_size] - cider_scores[batch_size:]
        scores+= cider_scores

    if use_ngram and opt.meteor_reward_weight > 0:
        _, meteor_scores = Meteor_scorer.compute_score(gts, res__)
        print('meteor_gen_score',np.mean(meteor_scores[:batch_size]), 'meteor_greedy_score', np.mean(meteor_scores[batch_size:]))

        meteor_scores = opt.meteor_reward_weight * np.array(meteor_scores)
        meteor_scores = meteor_scores[:batch_size] - meteor_scores[batch_size:]
        scores+= meteor_scores

    rewards = np.repeat(scores[:, np.newaxis], gen_result_data.shape[1], 1)
    return rewards

# def get_self_critical_reward_paragraph(gen_result, greedy_result, gen_d, greedy_d, gt_d, data, sent_num,opt):
#     batch_size = gen_result.size(0)
#
#     # Discriminator Score
#     scores = np.zeros(batch_size)
#     if opt.gan and opt.gan_reward_weight > 0 and opt.d_use_paragraph:
#         dis_scores = opt.gan_reward_weight * (gen_d - greedy_d)
#         scores += dis_scores.data.cpu().numpy()
#         print('dis_gen_par_score', gen_d.mean().item(), 'dis_greedy_par_score', greedy_d.mean().item(), 'dis_gt_par_score', gt_d.mean().item())
#
#     # Prep for NLP Metrics
#     res = OrderedDict()
#
#     gen_result_data = utils.combine_seq(sent_num,gen_result).data.cpu().numpy()
#     greedy_result_data = utils.combine_seq(sent_num,greedy_result).data.cpu().numpy()
#
#     for i in range(batch_size):
#         res[i] = [array_to_str(np.trim_zeros(gen_result_data[i]))] # remove 0's (eos token) to evaluate on paragraph level
#     for i in range(batch_size):
#         res[batch_size + i] = [array_to_str(np.trim_zeros(greedy_result_data[i]))]
#
#     gts = OrderedDict()
#     for i in range(len(data['gts'])):
#         gts[i] = [array_to_str(np.trim_zeros(data['gts'][i,:sent_num[i]].flatten()))]
#
#     res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)] # for cider
#     res__ = {i: res[i] for i in range(2 * batch_size)} # for bleu
#     gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
#
#     if opt.cider_reward_weight > 0:
#         _, cider_scores = CiderD_scorer.compute_score(gts, res_)
#         print('cider_p_gen_score',np.mean(cider_scores[:batch_size]), 'cider_p_greedy_score', np.mean(cider_scores[batch_size:]))
#
#         cider_scores = opt.cider_reward_weight * np.array(cider_scores)
#         cider_scores = cider_scores[:batch_size] - cider_scores[batch_size:]
#         scores+= cider_scores
#
#     if opt.meteor_reward_weight > 0:
#         _, meteor_scores = Meteor_scorer.compute_score(gts, res__)
#         print('p_meteor_gen_score',np.mean(meteor_scores[:batch_size]), 'p_meteor_greedy_score', np.mean(meteor_scores[batch_size:]))
#         meteor_scores = opt.meteor_reward_weight * np.array(meteor_scores)
#         meteor_scores = meteor_scores[:batch_size] - meteor_scores[batch_size:]
#         scores+= meteor_scores
#
#     rewards = np.repeat(scores[:, np.newaxis], gen_result_data.shape[1], 1)
#     return rewards
