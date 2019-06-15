from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import time
import json
import os

from six.moves import cPickle

import opts
import models
from dataloader import *
from train_utils import *
from eval_utils import eval_split
import misc.utils as utils

import gc

# try:
#     import tensorboardX as tb
# except ImportError:
#     print("tensorboardX is not installed")
#     tb = None

# There seems to be cpu memory leak in lstm?
# https://github.com/pytorch/pytorch/issues/3665
torch.backends.cudnn.enabled = False

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)
    if not os.path.exists(opt.checkpoint_path):
        os.mkdir(opt.checkpoint_path)

    with open(os.path.join(opt.checkpoint_path,'config.json'),'w') as f:
        json.dump(vars(opt),f)

    # Load iterators
    loader = DataLoader(opt)
    dis_loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.activity_size = loader.activity_size
    opt.seq_length = loader.seq_length
    opt.video = 1

    # set up models
    gen, dis = models.setup(opt)
    gen_model = gen.cuda()
    gen_model.train()
    dis_model = dis.cuda()
    dis_model.train()
    gen_optimizer = utils.build_optimizer(gen_model.parameters(), opt)
    dis_optimizer = utils.build_optimizer(dis_model.parameters(), opt)

    # loss functions
    crit = utils.LanguageModelCriterion()
    gan_crit = nn.BCELoss().cuda()

    # keep track of iteration
    g_iter = 0
    g_epoch = 0
    d_iter = 0
    d_epoch = 0
    dis_flag = False
    update_lr_flag = True

    # Load from checkpoint path
    infos = {'opt': opt}
    histories = {}
    infos['vocab'] = loader.get_vocab()
    if opt.g_start_from is not None:
        # Open old infos and check if models are compatible
        with open(os.path.join(opt.g_start_from, 'infos.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # Load train/val histories
        with open(os.path.join(opt.g_start_from, 'histories.pkl')) as f:
            histories = cPickle.load(f)

        # Load generator
        g_start_epoch = opt.g_start_epoch
        g_model_path = os.path.join(opt.g_start_from, "gen_%s.pth" % g_start_epoch)
        g_optimizer_path = os.path.join(opt.g_start_from, "gen_optimizer_%s.pth" % g_start_epoch)
        assert os.path.isfile(g_model_path) and os.path.isfile(g_optimizer_path)
        gen_model.load_state_dict(torch.load(g_model_path))
        gen_optimizer.load_state_dict(torch.load(g_optimizer_path))
        if "latest" not in g_start_epoch and "best" != g_start_epoch:
            g_epoch = int(g_start_epoch) + 1
            g_iter = (g_epoch) * loader.split_size['train'] // opt.batch_size
        else:
            g_epoch = infos['g_epoch_' + g_start_epoch] + 1
            g_iter = infos['g_iter_' + g_start_epoch]
        print('loaded %s (epoch: %d iter: %d)' % (g_model_path, g_epoch, g_iter))

        # Load discriminator
        # assume that discriminator is loaded only if generator has been trained and saved in the same directory.
        if opt.d_start_from is not None:
            d_start_epoch = opt.d_start_epoch
            d_model_path = os.path.join(opt.d_start_from, "dis_%s.pth" % d_start_epoch)
            d_optimizer_path = os.path.join(opt.d_start_from, "dis_optimizer_%s.pth" % d_start_epoch)
            assert os.path.isfile(d_model_path) and os.path.isfile(d_optimizer_path)
            dis_model.load_state_dict(torch.load(d_model_path))
            dis_optimizer.load_state_dict(torch.load(d_optimizer_path))
            if "latest" not in d_start_epoch and "best" != d_start_epoch:
                d_epoch = int(d_start_epoch) + 1
                d_iter = (d_epoch) * loader.split_size['train'] // opt.batch_size
            else:
                d_epoch = infos['d_epoch_' + d_start_epoch] + 1
                d_iter = infos['d_iter_' + d_start_epoch]
            print('loaded %s (epoch: %d iter: %d)' % (d_model_path, d_epoch, d_iter))
    infos['opt'] = opt
    loader.iterators = infos.get('g_iterators', loader.iterators)
    dis_loader.iterators = infos.get('d_iterators', loader.iterators)

    # hybrid discriminator weight
    v_weight = opt.visual_weight
    l_weight = opt.lang_weight
    p_weight = opt.par_weight

    # misc
    best_val_score = infos.get('g_best_score', None)
    best_d_val_score = infos.get('d_best_score', None)
    opt.activity_size = loader.activity_size
    opt.seq_length = loader.seq_length
    opt.video = 1
    g_val_result_history = histories.get('g_val_result_history', {})
    d_val_result_history = histories.get('d_val_result_history', {})
    g_loss_history = histories.get('g_loss_history', {})
    d_loss_history = histories.get('d_loss_history', {})

    """ START TRAINING """
    while True:
        gc.collect()
        # set every epoch
        if update_lr_flag:
            # Assign the learning rate for generator
            if g_epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (g_epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(gen_optimizer, opt.current_lr)

            # Assign the learning rate for discriminator
            if dis_flag and d_epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (d_epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(dis_optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if g_epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (g_epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                gen.ss_prob = opt.ss_prob

            # Start using previous sentence as context for generator (default: 10 epoch)
            if opt.g_context_epoch >= 0 and g_epoch >= opt.g_context_epoch:
                gen_model.use_context()

            # Switch to training discriminator
            if opt.g_pre_nepoch >= 0 and g_epoch >= opt.g_pre_nepoch and not dis_flag:
                print('Switching to pre-training discrimiator...')
                loader.reset_iterator('train')
                dis_loader.reset_iterator('train')
                dis_flag = True

            update_lr_flag = False

        """ TRAIN GENERATOR """
        if not dis_flag:
            gen_model.train()

            # train generator
            start = time.time()
            gen_loss, wrapped, sent_num = train_generator(gen_model, gen_optimizer, crit, loader)
            end = time.time()

            # Print Info
            if g_iter % opt.losses_print_every == 0:
                print("g_iter {} (g_epoch {}), gen_loss = {:.3f}, time/batch = {:.3f}, num_sent = {} {}" \
                    .format(g_iter, g_epoch, gen_loss, end - start,sum(sent_num),sent_num))

            # Log Losses
            if g_iter % opt.losses_log_every == 0:
                g_loss = gen_loss
                g_loss_history[g_iter] = {'g_loss': g_loss, 'g_epoch': g_epoch}

            # Update the iteration
            g_iter += 1

            #########################
            # Evaluate & Save Model #
            #########################
            if wrapped:
                # evaluate model on dev set
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json,
                               'sample_max' : 1,
                               'language_eval': opt.language_eval,
                               'id' : opt.id,
                               'val_videos_use' : opt.val_videos_use,
                               'remove' : 1} # remove generated caption
                # eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats, _, _ = eval_split(gen_model, crit, loader, eval_kwargs=eval_kwargs)
                if opt.language_eval == 1:
                    current_score = lang_stats['METEOR']
                else:
                    current_score = - val_loss
                g_val_result_history[g_epoch] = {'g_loss': val_loss, 'g_score': current_score, 'lang_stats': lang_stats}

                # Save the best generator model
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'gen_best.pth')
                    torch.save(gen_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'gen_optimizer_best.pth'))
                    infos['g_epoch_best'] = g_epoch
                    infos['g_best_score'] = best_val_score
                    torch.save(gen_model.state_dict(), checkpoint_path)
                    print("best generator saved to {}".format(checkpoint_path))

                # Dump miscalleous informations and save
                infos['g_epoch_latest'] = g_epoch
                infos['g_iter_latest'] = g_iter
                infos['g_iterators'] = loader.iterators
                histories['g_val_result_history'] = g_val_result_history
                histories['g_loss_history'] = g_loss_history
                with open(os.path.join(opt.checkpoint_path, 'infos.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                # save the latest model
                if opt.save_checkpoint_every > 0 and g_epoch % opt.save_checkpoint_every == 0:
                    torch.save(gen.state_dict(), os.path.join(opt.checkpoint_path, 'gen_%d.pth'% g_epoch))
                    torch.save(gen.state_dict(), os.path.join(opt.checkpoint_path, 'gen_latest.pth'))
                    torch.save(gen_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'gen_optimizer_%d.pth'% g_epoch))
                    torch.save(gen_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'gen_optimizer_latest.pth'))
                    print("model saved to {} at epoch {}".format(opt.checkpoint_path, g_epoch))

                # update epoch and lr
                g_epoch += 1
                update_lr_flag = True

        """ TRAIN DISCRIMINATOR """
        if dis_flag:
            dis_model.train()
            gen_model.eval()
            # choose negatives to use for visual discriminator
            if d_epoch >= 2 and d_iter % 2 == 0:
                dis_loader.set_negatives('hard')
            else:
                dis_loader.set_negatives('random')

            # set temperature
            if opt.dynamic_temperature:
                temp_range = [1.0, 0.8, 0.6, 0.4, 0.2]
                temperature = temp_range[d_iter % (len(temp_range))]
            else:
                temperature = opt.train_temperature

            # train discriminator
            start = time.time()
            losses, accuracies, wrapped,sent_num = train_discriminator(dis_model,gen_model,dis_optimizer,gan_crit,dis_loader,
                                                                       temperature=temperature,gen_weight=opt.d_gen_weight,mm_weight=opt.d_mm_weight,
                                                                       use_vis=(v_weight >0), use_lang=(l_weight > 0), use_pair=(p_weight>0))
            dis_v_loss, dis_l_loss, dis_p_loss = losses
            end = time.time()

            # Print Info
            if d_iter % opt.losses_print_every == 0:
                print("d_iter {} (d_epoch {}), v_loss = {:.8f}, l_loss = {:.8f}, p_loss={:.8f}, time/batch = {:.3f}, num_sent = {} {}" \
                    .format(d_iter, d_epoch, dis_v_loss, dis_l_loss, dis_p_loss, end - start,sum(sent_num),sent_num))
                print("accuracies:", accuracies)

            # Log Losses
            if d_iter % opt.losses_log_every == 0:
                d_loss_history[d_iter] = {'dis_v_loss': dis_v_loss, 'dis_l_loss': dis_l_loss, 'dis_p_loss': dis_p_loss, 'd_epoch': d_epoch}
                for type, accuracy in accuracies.items():
                    d_loss_history[d_iter][type] = accuracy

            # Update the iteration
            d_iter += 1

            #########################
            # Evaluate & Save Model #
            #########################
            if wrapped:
                # evaluate model on dev set
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json,
                               'sample_max' : (d_epoch+1) % 5 != 0,
                               'num_samples' : 30,
                               'temperature' : 0.2,
                               'language_eval' : opt.language_eval,
                               'id' : opt.id,
                               'val_videos_use': opt.val_videos_use,
                               'remove' : 1}
                _ , predictions, lang_stats, val_result, _ = eval_split(gen_model, crit, loader, dis_model, gan_crit,
                                                                        eval_kwargs=eval_kwargs)
                d_val_result_history[d_epoch] = val_result

                # save the best discriminator
                current_d_score = v_weight * (val_result['v_gen_accuracy'] + val_result['v_mm_accuracy']) + \
                                  l_weight  * (val_result['l_gen_accuracy'] + val_result['l_neg_accuracy']) + \
                                  p_weight * (val_result['p_gen_accuracy'] + val_result['p_neg_accuracy'])
                if best_d_val_score is None or current_d_score > best_d_val_score:
                    best_d_val_score = current_d_score
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'dis_best.pth')
                    torch.save(dis_model.state_dict(),checkpoint_path)
                    torch.save(dis_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'dis_optimizer_best.pth'))
                    infos['d_epoch_best'] = d_epoch
                    infos['d_iter_best'] = d_iter
                    infos['d_best_score'] = best_d_val_score
                    print("best discriminator saved to {}".format(checkpoint_path))

                # Dump miscalleous informations
                infos['d_epoch_latest'] = d_epoch
                infos['d_iter_latest'] = d_iter
                infos['d_iterators'] = dis_loader.iterators
                histories['d_loss_history'] = d_loss_history
                histories['d_val_result_history'] = d_val_result_history
                with open(os.path.join(opt.checkpoint_path, 'infos.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                # save model
                if opt.save_checkpoint_every > 0 and d_epoch % opt.save_checkpoint_every == 0:
                    torch.save(dis.state_dict(), os.path.join(opt.checkpoint_path, 'dis_%d.pth'% d_epoch))
                    torch.save(dis.state_dict(), os.path.join(opt.checkpoint_path, 'dis_latest.pth'))
                    torch.save(dis_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'dis_optimizer_%d.pth'% d_epoch))
                    torch.save(dis_optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'dis_optimizer_latest.pth'))

                # update epoch and lr
                d_epoch += 1
                update_lr_flag = True

if __name__ == '__main__':
    opt = opts.parse_opt()
    train(opt)
