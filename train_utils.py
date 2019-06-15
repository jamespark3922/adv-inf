import torch
import misc.utils as utils

def train_generator(gen_model, gen_optimizer, crit, loader, grad_clip=0.1):

    data = loader.get_batch('train')
    torch.cuda.synchronize()
    tmp = [data['fc_feats'], data['att_feats'], data['img_feats'], data['box_feats'],
           data['labels'], data['masks'], data['att_masks'], data['activities']]
    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    fc_feats, att_feats, img_feats, box_feats, labels, masks, att_masks, activities = tmp
    sent_num = data['sent_num']
    wrapped = data['bounds']['wrapped']
    gen_optimizer.zero_grad()

    seq = gen_model(fc_feats, img_feats, box_feats, activities, labels)
    seq = utils.align_seq(sent_num, seq)
    labels = utils.align_seq(sent_num, labels)
    masks = utils.align_seq(sent_num, masks)
    loss = crit(seq, labels[:, 1:], masks[:, 1:])
    loss.backward()
    gen_loss = loss.item()

    utils.clip_gradient(gen_optimizer, grad_clip)
    gen_optimizer.step()
    torch.cuda.synchronize()

    return gen_loss, wrapped, sent_num

def train_discriminator(dis_model, gen_model, dis_optimizer, gan_crit, loader,
                        temperature=1.0,gen_weight=0.5, mm_weight=0.5,neg_weight=0.5,
                        use_vis=True,use_lang=True,use_pair=True,grad_clip=0.1):
    dis_model.train()
    gen_model.eval()
    data = loader.get_batch('train')
    sent_num = data['sent_num']
    torch.cuda.synchronize()
    tmp = [data['fc_feats'],data['mm_fc_feats'], data['img_feats'], data['box_feats'], data['att_feats'], data['labels'], data['mm_labels'],
           data['att_masks'], data['activities'], data['mm_img_feats'], data['mm_box_feats'], data['mm_activities']]
    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    fc_feats, mm_fc_feats, img_feats, box_feats, att_feats, labels, mm_labels, att_masks, activities, \
    mm_img_feats, mm_box_feats, mm_activities = tmp
    label = torch.zeros(sum(sent_num)).cuda()
    dis_v_loss = 0
    dis_l_loss = 0
    dis_p_loss = 0
    accuracies = {}
    wrapped = data['bounds']['wrapped']

    with torch.no_grad():

        # generated captions
        gen_labels, sample_logprobs = gen_model(fc_feats, img_feats, box_feats, activities,
                                                opt={'sample_max':0,'temperature':temperature}, mode='sample')
        masks = utils.generate_paragraph_mask(sent_num,gen_labels)
        gen_labels = torch.mul(gen_labels, masks)

        # visually mismatched negatives from generator
        mm_gen_labels, mm_sample_logprobs = gen_model(mm_fc_feats, mm_img_feats, mm_box_feats, mm_activities,
                                                      opt={'sample_max': 0, 'temperature': temperature}, mode='sample')
        mm_masks = utils.generate_paragraph_mask(sent_num,mm_gen_labels)
        mm_gen_labels = torch.mul(mm_gen_labels, mm_masks)

        # gen or gt sentence as language negatives
        neg_lang_labels = utils.get_neg_lang(sent_num,labels,gen_labels)

        # only gt sentence pair as pairwise negatives
        neg_pair_labels = torch.from_numpy(utils.get_neg_pair(sent_num, data['labels'])).cuda()

    # update visual discriminator with [gt (real), gt mismatch (fake), gen mismatch (fake)]
    if use_vis:
        dis_optimizer.zero_grad()

        # mismatch_gt
        v_mm_score = dis_model(fc_feats, img_feats, box_feats, activities, mm_labels[:, :, 1:-1])
        v_mm_score = utils.align_seq(sent_num, v_mm_score)
        v_loss_3 = mm_weight * gan_crit(v_mm_score, label)
        v_loss_3.backward()
        dis_v_loss += v_loss_3.item()

        # mismatch_gen
        v_mm_gen_score = dis_model(fc_feats, img_feats, box_feats, activities, mm_gen_labels)
        v_mm_gen_score = utils.align_seq(sent_num, v_mm_gen_score)
        v_loss_1 = gen_weight * gan_crit(v_mm_gen_score, label)
        v_loss_1.backward()
        dis_v_loss += v_loss_1.item()

        # gt
        label.fill_(1)
        v_gt_score = dis_model(fc_feats, img_feats, box_feats, activities, labels[:, :, 1:-1])
        v_gt_score = utils.align_seq(sent_num, v_gt_score)
        v_loss_2 = gan_crit(v_gt_score, label)
        v_loss_2.backward()
        dis_v_loss += v_loss_2.item()

        # update discriminator
        utils.clip_gradient(dis_optimizer, grad_clip)
        dis_optimizer.step()
        torch.cuda.synchronize()

    # update language discriminator with [gt (real), gen(fake), neg (fake)]
    if use_lang:
        dis_optimizer.zero_grad()

        # gen
        label.fill_(0)
        l_gen_score = dis_model(gen_labels, mode='lang')
        l_gen_score = utils.align_seq(sent_num, l_gen_score)
        l_loss_1 = gan_crit(l_gen_score, label)
        l_loss_1.backward()
        dis_l_loss += l_loss_1.item()

        # negative sample
        l_neg_score = dis_model(neg_lang_labels, mode='lang')
        l_neg_score = utils.align_seq(sent_num, l_neg_score)
        l_loss_3 = neg_weight * gan_crit(l_neg_score, label)
        l_loss_3.backward()
        dis_l_loss += l_loss_3.item()

        # gt
        label.fill_(1)
        l_gt_score = dis_model(labels[:, :, 1:-1], mode='lang')
        l_gt_score = utils.align_seq(sent_num, l_gt_score)
        l_loss_2 = gan_crit(l_gt_score, label)
        l_loss_2.backward()
        dis_l_loss += l_loss_2.item()

        # update discriminator
        utils.clip_gradient(dis_optimizer, grad_clip)
        dis_optimizer.step()
        torch.cuda.synchronize()

    # update pairwise discriminator [gt (real), gen(fake), neg(fake)]
    if use_pair:
        dis_optimizer.zero_grad()
        # gen
        label.fill_(0)
        p_gen_score = dis_model(gen_labels, mode='par')
        p_gen_score = utils.align_seq(sent_num, p_gen_score)
        p_loss_1 = gan_crit(p_gen_score, label)
        p_loss_1.backward()
        dis_p_loss += p_loss_1.item()

        # negative sample
        p_neg_score = dis_model( neg_pair_labels[:,:,1:-1], mode='par')
        p_neg_score = utils.align_seq(sent_num, p_neg_score)
        p_loss_3 = neg_weight * gan_crit(p_neg_score, label)
        p_loss_3.backward()
        dis_p_loss += p_loss_3.item()

        # gt
        label.fill_(1)
        s = 0
        for n in sent_num:
            label[s] = 0 # first sentence is assigned score 0
            s+=n
        p_gt_score = dis_model(labels[:, :, 1:-1], mode='par')
        p_gt_score = utils.align_seq(sent_num, p_gt_score)
        p_loss_2 = gan_crit(p_gt_score, label)
        p_loss_2.backward()
        dis_p_loss += p_loss_2.item()

        # update discriminator
        utils.clip_gradient(dis_optimizer, grad_clip)
        dis_optimizer.step()
        torch.cuda.synchronize()

    # calculate accuracy (ground truth scores higher than negative inputs)
    with torch.no_grad():
        if use_vis:
            v_gen_accuracy = torch.gt(v_gt_score, v_mm_gen_score).cpu().numpy().mean()
            v_mm_accuracy = torch.gt(v_gt_score, v_mm_score).cpu().numpy().mean()
            accuracies['dis_v_gen_accuracy'] = v_gen_accuracy
            accuracies['dis_v_mm_accuracy'] = v_mm_accuracy
        if use_lang:
            l_gen_accuracy = torch.gt(l_gt_score, l_gen_score).cpu().numpy().mean()
            l_neg_accuracy = torch.gt(l_gt_score, l_neg_score).cpu().numpy().mean()
            accuracies['dis_l_gen_accuracy'] = l_gen_accuracy
            accuracies['dis_l_neg_accuracy'] = l_neg_accuracy
        if use_pair:
            p_gen_accuracy = torch.gt(p_gt_score, p_gen_score).cpu().numpy().mean()
            p_neg_accuracy = torch.gt(p_gt_score, p_neg_score).cpu().numpy().mean()
            accuracies['dis_p_gen_accuracy'] = p_gen_accuracy
            accuracies['dis_p_neg_accuracy'] = p_neg_accuracy

    return [dis_v_loss, dis_l_loss, dis_p_loss], accuracies, wrapped, sent_num