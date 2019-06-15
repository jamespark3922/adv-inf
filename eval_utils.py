from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
import string
import random
import shutil
import os
import sys
import misc.utils as utils
import subprocess
from six.moves import cPickle
import time

def extend_paragraph(sent_num,par_score):
    new_score = par_score.new(sum(sent_num)).zero_()
    m = 0
    for i,n in enumerate(sent_num):
        for j in range(n):
            new_score[m+j:m+j+1] = par_score[i]
        m+=n
    return new_score

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def language_eval_video(dataset, preds, model_id, split, verbose=False, remove=False):
    import sys
    sys.path.append("densevid_eval")
    template = {"version": "VERSION 1.0", "results": {},
                "external_data": { "used": 'true',
            "details": "ay"}  }
    results = template['results']
    for pred in preds:
        id = pred['video_id']
        sent = ' '.join([word for word in pred['caption'].split() if word != '<unk>'])
        info = {'sentence': sent,  # String description of an event.
                'timestamp' : pred['timestamp'],
                'activity' : pred['activity']} # The start and end times of the event (in seconds).
        if id not in results:
            results[id] = []
        results[id].append(info)
    if remove:
        model_id += id_generator() # to avoid processing and removing same ids
    json.dump(template, open(os.path.join('densevid_eval', 'caption_' + model_id + '.json'), 'w'))
    eval_command = ["python","para-evaluate.py", "-s",'caption_' + model_id + '.json',
                    "-o", 'result_' + model_id + '.json', '--verbose']
    subprocess.call(eval_command,cwd='densevid_eval')
    output = json.load(open(os.path.join('densevid_eval','result_' + model_id + '.json'),'r'))
    if remove:
        os.remove(os.path.join('densevid_eval','caption_' + model_id + '.json'))
        os.remove(os.path.join('densevid_eval','result_' + model_id + '.json'))
    return output

def bigram(sent):
    return zip(sent.split(" ")[:-1], sent.split(" ")[1:])

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
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
        out.append(txt)
    return out

def diversity_meausures(predictions,div):
    vocab = {'gt': set(), 'gen': set()}
    sentences = {'gt' : {'total': [], 'unique': set()} , 'gen': {'total': [], 'unique': set()} }
    length = {'gt': [], 'gen': []}
    vocab_5 = {'gt' : set(), 'gen': set() }
    sentences_5 = {'gt' : {'total': [], 'unique': set()} , 'gen': {'total': [], 'unique': set()} }

    div_1 = {'gt' : [], 'gen': []}
    div_2 = {'gt' : [], 'gen': []}

    template = {'vocab_size' : {}, 'novel_sentences' : {} , 'sent_length': {}}

    for entry in predictions:
        for mode in ['gen', 'gt']:
            sent = entry['caption'] if mode == 'gen' else entry['gt']
            vocab[mode]|= set(sent.split())
            sentences[mode]['total'].append(sent)
            sentences[mode]['unique'].add(sent)
            length[mode].append(len(sent.split()))

    for mode in ['gen','gt']:
        template['vocab_size'][mode] = len(vocab[mode])
        template['novel_sentences'][mode] = round(len(sentences[mode]['unique']) / len(sentences[mode]['total']),3)
        template['sent_length'][mode] = np.mean(length[mode])

    for k in range(len(div['gen'])):
        for mode in ['gen','gt']:
            caption_list = div[mode][k]['captions'] # list of captions per image
            unigrams = [word for g in caption_list for word in g.split()]
            vocab_5[mode]|= set(unigrams)
            sentences_5[mode]['total'].extend(caption_list)
            sentences_5[mode]['unique']|= set(caption_list)
            div_1[mode].append(len(set(unigrams)) / len(unigrams))

            bigrams = [bg for g in caption_list for bg in bigram(g)]
            div_2[mode].append(len(set(bigrams)) / len(bigrams))

    if len(div_1['gen']) > 0: # diversity score for multiple captions
        for keys in ['vocab_size_5','novel_sentences_5','div_1','div_2']:
            template[keys] = {}
        for mode in ['gen','gt']:
            template['vocab_size_5'][mode] = len(vocab_5[mode])
            template['novel_sentences_5'][mode] = round(len(sentences_5[mode]['unique']) / len(sentences_5[mode]['total']),3)
            template['div_1'][mode] = round(np.mean(div_1[mode]),3)
            template['div_2'][mode] = round(np.mean(div_2[mode]),3)

    return template

def eval_split(gen_model, crit, loader, dis_model=None, gan_crit=None, classifier=None, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    dump_json = eval_kwargs.get('dump_json', 0)
    num_videos = eval_kwargs.get('num_videos', eval_kwargs.get('val_videos_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    use_context = eval_kwargs.get('use_context', 0)

    sample_max = eval_kwargs.get('sample_max', 1)
    beam_size = eval_kwargs.get('beam_size', 1)
    num_samples = eval_kwargs.get('num_samples', 100)
    num_captions = eval_kwargs.get('num_captions', 1)
    verbose_video = eval_kwargs.get('verbose_video', 0)
    remove_caption = eval_kwargs.get('remove', 0) # usually remove captions in validation stage but not in test.

    print('beam_size', beam_size)
    print('sample_max',sample_max)
    print('num_samples', num_samples)

    model_id = eval_kwargs.get('id', eval_kwargs.get('val_id', ''))

    if split == 'val':
        model_id = 'val_' + model_id

    if sample_max:
        assert num_captions <= beam_size
    else:
        assert num_captions <= num_samples

    if use_context:
        gen_model.use_context()
    # Make sure in the evaluation mode
    gen_model.eval()

    loader.reset_iterator(split)

    n = 0
    losses = []
    predictions = []

    vis_weight = eval_kwargs.get('vis_weight', 0.8)
    lang_weight = eval_kwargs.get('lang_weight', 0.2)
    pair_weight = eval_kwargs.get('pair_weight', 1.0)

    div = {'gt': [], 'gen': []}
    dis = dis_model is not None
    if dis:
        assert gan_crit is not None
        dis_model.eval()
        scores = {'v_gen_scores' : [], 'v_gt_scores' : [], 'v_mm_scores' : [], 'v_mm_gen_scores' : [],
                  'l_gen_scores' : [], 'l_gt_scores' : [], 'l_neg_scores': [],
                  'p_gen_scores' : [], 'p_gt_scores' : [], 'p_neg_scores': []}
        v_gen_accuracy = []
        v_mm_accuracy = []
        l_gen_accuracy = []
        l_neg_accuracy = []
        p_gen_accuracy = []
        p_neg_accuracy = []

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'], data['img_feats'], data['box_feats'], data['mm_fc_feats'], data['att_feats'], data['labels'], data['mm_labels'],
               data['masks'], data['att_masks'], data['activities'], data['mm_img_feats'], data['mm_box_feats'], data['mm_activities']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, img_feats, box_feats, mm_fc_feats, att_feats, labels, mm_labels, masks, att_masks, activities, \
        mm_img_feats, mm_box_feats, mm_activities = tmp
        sent_num = data['sent_num']

        torch.manual_seed(1234)

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if classifier is not None:
                activities = utils.dense_classifier(sent_num, fc_feats, img_feats, classifier)
                mm_activities = utils.dense_classifier(sent_num, mm_fc_feats, mm_img_feats, classifier)

            # calculate loss
            gen_seq = gen_model(fc_feats, img_feats, box_feats, activities, labels)
            gen_seq = utils.align_seq(sent_num, gen_seq)
            loss = crit(gen_seq, utils.align_seq(sent_num, labels)[:, 1:], utils.align_seq(sent_num, masks)[:, 1:]).item()
            losses.append(loss)

            # use greedy max for inference
            if sample_max:
                eval_kwargs['sample_max'] = 1
                seq, _ = gen_model(fc_feats, img_feats, box_feats, activities,
                               opt=eval_kwargs, mode='sample')

            # use sampling for inference
            else:
                sample_list = np.zeros((loader.batch_size, num_samples, loader.seq_length))
                context_list = np.zeros((loader.batch_size, num_samples, 512))
                seq_dummy = torch.zeros(loader.batch_size, 10, loader.seq_length).cuda()
                best_context = None
                best_seq = None
                for s in range(max(sent_num)):
                    v_score_list = np.zeros((loader.batch_size, num_samples))
                    l_score_list = np.zeros((loader.batch_size, num_samples))
                    p_score_list = np.zeros((loader.batch_size, num_samples))
                    prob_score_list = np.zeros((loader.batch_size, num_samples))
                    score_list = np.zeros((loader.batch_size, num_samples))
                    for i in range(num_samples):
                        fc_feats_s = fc_feats[:, s]
                        img_feats_s = img_feats[:, s]
                        box_feats_s = box_feats[:, s]
                        start = time.time()
                        seq, logprobs, context = gen_model.sample_sequential(fc_feats_s, img_feats_s, box_feats_s, activities,
                                                                             best_context, opt=eval_kwargs)
                        sample_time = time.time()
                        # print('sample_time:', sample_time-start)

                        """ Adversarial Inference """
                        if dis:
                            v_score = dis_model(fc_feats_s.unsqueeze(1), img_feats_s.unsqueeze(1), box_feats_s.unsqueeze(1),
                                                activities, seq.unsqueeze(1)).squeeze()
                            v_score_list[:, i] = v_score

                            vis_time = time.time()
                           # print('vis_time:', vis_time - sample_time)

                            l_score = dis_model(seq.unsqueeze(1), mode='lang').squeeze()
                            l_score_list[:,i] = l_score

                            lang_time = time.time()
                            # print('lang_time:', lang_time - vis_time)

                            if pair_weight > 0 and best_seq is not None:
                                pair_seq = torch.cat((best_seq.unsqueeze(1), seq.unsqueeze(1)), dim=1)
                                p_score = dis_model(pair_seq, mode='par')[:,1].squeeze()
                                p_score_list[:, i] = p_score

                                pair_time = time.time()
                                # print('pair_time:', pair_time - lang_time)

                            score_list[:,i] = vis_weight * v_score_list[:, i] + lang_weight * l_score_list[:, i] + pair_weight * p_score_list[:,i]

                        sample_list[:, i] = seq.cpu().numpy()
                        context_list[:, i] = context.squeeze(1)

                        prob_score = (torch.sum(logprobs, 1).cpu().numpy()) / np.count_nonzero(seq, axis=1)
                        prob_score_list[:, i] += prob_score
                        if score_list[:, i].sum() == 0:
                            score_list[:, i] += 0.5 * prob_score

                    # select the caption with highest score
                    inds = score_list.argsort(axis=1)[:, ::-1]
                    caption_list = torch.tensor(
                        sample_list[np.arange(loader.batch_size)[:, None], inds]).cuda().long()
                    best_context = torch.tensor(
                        context_list[np.arange(loader.batch_size)[:, None], inds][:, :1, :]).cuda().float()
                    best_seq = caption_list[:, 0, :]
                    seq_dummy[:, s] = best_seq

                # generated sequence
                seq = seq_dummy.long()

            # calculate discriminator scores for each input.
            if dis:
                seq = torch.mul(seq,utils.generate_paragraph_mask(sent_num, seq))

                # negatives for evaluating discriminator
                mm_seq, _ = gen_model(mm_fc_feats, mm_img_feats, mm_box_feats, mm_activities,
                                opt=eval_kwargs, mode='sample')
                mm_seq = torch.mul(mm_seq,utils.generate_paragraph_mask(sent_num, mm_seq))

                neg_lang_labels = utils.get_neg_lang(sent_num, labels, seq.cuda())
                neg_pair_labels = torch.from_numpy(utils.get_neg_pair(sent_num, data['labels'])).cuda()

                dis_loss = 0

                v_gen_score = dis_model(fc_feats, img_feats, box_feats, activities, seq.cuda())
                v_gen_score = utils.align_seq(sent_num, v_gen_score)
                l_gen_score = dis_model(seq.cuda(), mode='lang')
                l_gen_score = utils.align_seq(sent_num, l_gen_score)
                p_gen_score = dis_model(seq.cuda(), mode='par')
                p_gen_score = utils.align_seq(sent_num,p_gen_score)
                scores['v_gen_scores'].extend(v_gen_score)
                scores['l_gen_scores'].extend(l_gen_score)
                scores['p_gen_scores'].extend(p_gen_score)


                v_gt_score = dis_model(fc_feats, img_feats, box_feats, activities, labels[:,:,1:-1])
                v_gt_score = utils.align_seq(sent_num, v_gt_score)
                l_gt_score = dis_model(labels[:,:,1:-1], mode='lang')
                l_gt_score = utils.align_seq(sent_num, l_gt_score)
                p_gt_score = dis_model(labels[:,:,1:-1], mode='par')
                p_gt_score = utils.align_seq(sent_num, p_gt_score)
                scores['v_gt_scores'].extend(v_gt_score)
                scores['l_gt_scores'].extend(l_gt_score)
                scores['p_gt_scores'].extend(p_gt_score)


                v_mm_score = dis_model(fc_feats, img_feats, box_feats, activities, mm_labels[:,:,1:-1])
                v_mm_score = utils.align_seq(sent_num, v_mm_score)
                v_mm_gen_score = dis_model(fc_feats, img_feats, box_feats, activities, mm_seq.cuda())
                v_mm_gen_score = utils.align_seq(sent_num, v_mm_gen_score)
                l_neg_score = dis_model(neg_lang_labels, mode='lang')
                l_neg_score = utils.align_seq(sent_num, l_neg_score)
                p_neg_score = dis_model(neg_pair_labels, mode='par')
                p_neg_score = utils.align_seq(sent_num, p_neg_score)
                scores['v_mm_scores'].extend(v_mm_score)
                scores['v_mm_gen_scores'].extend(v_mm_gen_score)
                scores['l_neg_scores'].extend(l_neg_score)
                scores['p_neg_scores'].extend(p_neg_score)


            seq = utils.align_seq(sent_num,seq)
            labels = utils.align_seq(sent_num,labels)
            mm_labels = utils.align_seq(sent_num, mm_labels)
            gt = utils.decode_sequence(loader.get_vocab(),labels[:,1:-1].data)
            mm = utils.decode_sequence(loader.get_vocab(), mm_labels[:,1:-1].data)
            seq = seq.data

        # print and store actual decoded sentence
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            entry = {'video_id': data['infos'][k]['id'], 'caption': sent.encode('ascii', 'ignore'),
                     'gt' : gt[k].encode('ascii','ignore'), 'mm' : mm[k].encode('ascii','ignore'),
                    'timestamp': data['infos'][k]['timestamp'].tolist(),
                     'activity' : data['infos'][k]['activity']
                     }
            # calculate accuracy
            if dis:
                entry['v_gen_score'] = v_gen_score[k].item()
                entry['v_gt_score'] = v_gt_score[k].item()
                entry['v_mm_score'] = v_mm_score[k].item()
                entry['v_mm_gen_score'] = v_mm_gen_score[k].item()
                ga = 0
                ma = 0
                if entry['v_gt_score'] > entry['v_mm_gen_score']:
                    ga = 1
                if entry['v_gt_score'] > entry['v_mm_score']:
                    ma = 1
                v_gen_accuracy.append(ga)
                v_mm_accuracy.append(ma)

                entry['l_gen_score'] = l_gen_score[k].item()
                entry['l_gt_score'] = l_gt_score[k].item()
                ga = 0
                na = 0
                if entry['l_gt_score'] > entry['l_gen_score']:
                    ga = 1
                if entry['l_gt_score'] > l_neg_score[k].item():
                    na = 1
                l_gen_accuracy.append(ga)
                l_neg_accuracy.append(na)

                entry['p_gen_score'] = p_gen_score[k].item()
                entry['p_gt_score'] = p_gt_score[k].item()
                ga = 0
                na = 0
                if entry['p_gt_score'] > entry['p_gen_score']:
                    ga = 1
                if entry['p_gt_score'] > p_neg_score[k].item():
                    na = 1

                # only add if there was no change e.g. ignore first sentence
                if entry['p_gt_score'] != entry['p_gen_score']:
                    p_gen_accuracy.append(ga)
                if entry['p_gt_score'] != p_neg_score[k].item():
                    p_neg_accuracy.append(na)

            predictions.append(entry)

            if verbose:
                if dis:
                    print_str = 'video %s: activity: %s; caption: %s; gt: %s; mm: %s; v_gen_score: %5f; v_gt_score: %5f; v_mm_score %5f' \
                                % (entry['video_id'], entry['activity'], entry['caption'], entry['gt'], entry['mm'], entry['v_gen_score'], entry['v_gt_score'], entry['v_mm_score'])
                    print_str = '%s; l_gen_score: %5f; l_gt_score: %5f; p_gen_score: %5f; p_gt_score: %5f;' \
                                % (print_str, entry['l_gen_score'], entry['l_gt_score'], entry['p_gen_score'], entry['p_gt_score'])
                    print(print_str)
                else:
                    print('video %s: activity: %s; caption: %s; gt: %s' %(entry['video_id'], entry['activity'], entry['caption'], entry['gt']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_videos != -1:
            ix1 = min(ix1, num_videos)
        i = 0
        img_id = predictions[-1]['video_id']
        while i < (n-ix1):
            predictions.pop()
            if dis:
                v_gen_accuracy.pop()
                v_mm_accuracy.pop()
                l_gen_accuracy.pop()
            cur_id = predictions[-1]['video_id']
            if cur_id != img_id:
                i+=1
                img_id = cur_id

        if verbose:
            if dis:
                print('evaluating validation preformance... %d/%d gen: (%f) dis: (%f)' %(ix0 - 1, ix1, loss, dis_loss))
            else:
                print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))


        if data['bounds']['wrapped']:
            break
        if num_videos >= 0 and n >= num_videos:
            break

    # Switch back to training mode
    gen_model.train()

    # calculate language metrics score
    gen_loss = np.mean(losses)
    lang_stats = None
    if lang_eval == 1:
        diversity_dict = diversity_meausures(predictions,div)
        lang_stats = language_eval_video(dataset, predictions, model_id, split , verbose=verbose_video, remove=remove_caption)
        lang_stats.update(diversity_dict)
        lang_stats.update({'loss': gen_loss})
        print(lang_stats)

    # discriminator accuracies and score stats for each input
    dis_infos = {}
    if dis:
        dis_infos['v_gen_accuracy'] = np.mean(v_gen_accuracy)
        dis_infos['v_mm_accuracy'] = np.mean(v_mm_accuracy)
        for mode in ['gen', 'gt', 'mm']:
            dis_infos['v_%s_avg' % mode] = np.mean(scores['v_%s_scores' % mode])
            dis_infos['v_%s_std' % mode] = np.std(scores['v_%s_scores' % mode])

        dis_infos['l_gen_accuracy'] = np.mean(l_gen_accuracy)
        dis_infos['l_neg_accuracy'] = np.mean(l_neg_accuracy)
        for mode in ['gen', 'gt', 'neg']:
            dis_infos['l_%s_avg' % mode] = np.mean(scores['l_%s_scores' % mode])
            dis_infos['l_%s_std' % mode] = np.std(scores['l_%s_scores' % mode])

        dis_infos['p_gen_accuracy'] = np.mean(p_gen_accuracy)
        dis_infos['p_neg_accuracy'] = np.mean(p_neg_accuracy)
        for mode in ['gen', 'gt', 'neg']:
            dis_infos['p_%s_avg' % mode] = np.mean(scores['p_%s_scores' % mode])
            dis_infos['p_%s_std' % mode] = np.std(scores['p_%s_scores' % mode])

        print(sorted(dis_infos.items()))

    if dump_json == 1:
        # dump the json
        json.dump(lang_stats, open('eval_results/' + model_id + '.json', 'w'))
        json.dump(predictions, open('vis/vis_' + model_id + '.json', 'w'))
        json.dump(div['gen'], open('vis/vis_n_' + model_id + '.json', 'w'))

    return gen_loss, predictions, lang_stats, dis_infos, div
