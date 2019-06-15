import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/video_data_dense_label.json',
                    help='path to the json file containing additional info and vocab (img/video)')
    parser.add_argument('--input_fc_dir', type=str,
                        help='path to the directory containing the preprocessed fc video features')
    parser.add_argument('--input_img_dir', type=str,
                        help='path to the directory containing the image features')
    parser.add_argument('--input_box_dir', type=str,
                    help='path to the directory containing the boxes of att img feats (img)')
    parser.add_argument('--input_label_h5', type=str, default='data/video_data_dense_label.h5',
                    help='path to the h5file containing the preprocessed dataset (img/video)')

    parser.add_argument('--g_start_from', type=str, default=None,
                     help="""skip pre training step and continue training from saved generator model at this path.
                          'infos_{id}.pkl'         : configuration;
                          'gen_optimizer_{epoch}.pth'     : optimizer;
                          'gen_{epoch}.pth'         : model
                     """)
    parser.add_argument('--g_start_epoch', type=str, default="latest",
                     help="""start training generator at epoch (int, latest, latest_ce, latest_scst)
                     """)
    parser.add_argument('--d_start_from', type=str, default=None,
                    help="""skip pre training step and continue training from saved discrimiator model at this path.
                          for now, assumes that generator has been loaded as well. (Note generator's infos.pkl will be used)
                          'infos_{id}.pkl'         : configuration;
                          'dis_optimizer_{epoch}.pth'     : optimizer;
                          'dis_{epoch}.pth'         : model
                    """)
    parser.add_argument('--d_start_epoch', type=str, default="latest",
                     help="""start training discriminator at epoch (int, latest)
                     """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="video",
                    help='fc, show_tell, adaatt, topdown, s2vt, paragraph show_attend_tell, all_img, att2in, att2in2, att2all2,  stackatt, denseatt')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--d_rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--video_encoding_size', type=int, default=256,
                    help='the encoding size of each frame of c3d features.')
    parser.add_argument('--d_video_encoding_size', type=int, default=256,
                    help='the encoding size of each frame of c3d features.')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--d_input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=8192,
                    help='2048 for resnet, 4096 for vgg (img) \
                          500  for c3d,    8192 for r3d (video')
    parser.add_argument('--img_feat_size', type=int, default=2048,
                        help='img feat size')
    parser.add_argument('--box_feat_size', type=int, default=8473,
                        help='box feat size')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')

    # input settings
    parser.add_argument('--use_video', type=int, default=1,
                        help='use video features (c3d/resnext101-64f) specified in input_fc_dir')
    parser.add_argument('--d_use_video', type=int, default=1,
                        help='use video features (c3d/resnext101-64f) specified in input_fc_dir for discriminator')
    parser.add_argument('--use_img', type=int, default=1,
                        help='use resnet features specified in input_img_dir')
    parser.add_argument('--d_use_img', type=int, default=1,
                        help='use resnet features specified in input_img_dir for discriminator')
    parser.add_argument('--use_box', type=int, default=1,
                        help='use bottomup features sepcified in input_box_dir')
    parser.add_argument('--d_use_box', type=int, default=1,
                        help='use bottomup features sepcified in input_box_dir for discriminator')
    parser.add_argument('--d_use_bow', type=int, default=1,
                        help='use bag of words for visual discriminator; otherwise, use lstm')
    parser.add_argument('--glove_npy', type=str, default=None,
                        help='npy containing glove vector associated with word_idx labels')

    # video options
    parser.add_argument('--feat_type', type=str, default='resnext101-64f',
                        help='feat type for video (c3d, resnext101-64f)')
    parser.add_argument('--g_context_epoch', type=int, default=10,
                        help='epoch to start incorporating context for generator (-1 = dont use context)')
    parser.add_argument('--d_context_epoch', type=int, default=0,
                        help='epoch to start incorporating context for discriminator (-1 = dont use context)')
    parser.add_argument('--use_activity_labels', type=int, default=0,
                        help='make captioning model use activity label as additional feature')
    parser.add_argument('--d_use_activity_labels', type=int, default=0,
                        help='make discriminator use activity label as additional feature')
    parser.add_argument('--activity_encoding_size', type=int, default=50,
                        help='encoding size of activity labels to feed into rnn. should set lte activity label size (200).')
    parser.add_argument('--context_encoding_size', type=int, default=50,
                        help='size to encode last hidden lstm state')
    parser.add_argument('--use_paragraph', type=int, default=0,
                        help='calculate paragraph level score for generator')
    parser.add_argument('--d_use_paragraph', type=int, default=0,
                        help='calculate paragraph level score from discriminator')
    parser.add_argument('--negatives', type=str, default='random',
                        help='option for mismatched (video,caption) pair for discriminator. \
                              random: random caption      \
                              hard: different video with same activity')

    # video disc option
    parser.add_argument('--visual_weight', type=float, default=1.0,
                        help='weight to visual discriminator reward')
    parser.add_argument('--lang_weight', type=float, default=1.0,
                        help='weight to lang discriminator reward')
    parser.add_argument('--par_weight', type=float, default=1.0,
                        help='weight to paragraph discriminator reward')
    parser.add_argument('--ce_weight', type=float, default=0,
                        help='add ce loss during self-critical training')

    # classifier option
    parser.add_argument('--use_mean', type=int, default=0)
    parser.add_argument('--random_nfeat', type=int, default=20)
    parser.add_argument('--max_seg', type=int, default=10)
    parser.add_argument('--box_seg', type=int, default=3)

    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    # parser.add_argument('--use_box', type=int, default=0,
    #                 help='If use box features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')

    # Optimization: General
    parser.add_argument('--g_pre_nepoch', type=int, default=50,
                    help='number of epochs to pre-train generator with cross entropy')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_id', type=str, default='',
                        help='id to use to save captions for validation')
    parser.add_argument('--val_videos_use', type=int, default=-1,
                    help='how many videos to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--losses_print_every', type=int, default=10,
                    help='How often do we want to print losses? (0 = disable)')
    parser.add_argument('--save_checkpoint_every', type=int, default=1,
                    help='how often to save a model checkpoint in iterations? the code already saves checkpoint every epoch (0 = dont save; 1 = every epoch)')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    # Sampling
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
    parser.add_argument('--train_temperature', type=float, default=1.0,
                        help='temperature when sampling from distributions to train the discriminator.')
    parser.add_argument('--dynamic_temperature', action='store_true',
                        help='use temperature from range [1.0, 0.8, 0.6, 0.4, 0.2] alternatively')

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=0,
                    help='The reward weight from cider')
    parser.add_argument('--gan_reward_weight', type=float, default=1,
                    help='The reward weight from gan')
    parser.add_argument('--meteor_reward_weight', type=float, default=0,
                    help='The reward weight from meteor')

    # Discriminator
    parser.add_argument('--dis_model', type=str, default="joint_embed",
                    help='joint_embed, co_att, fc, fc_video, s2vt')
    parser.add_argument('--d_pre_nepoch', type=int, default=10,
                    help='number of epochs to pre-train discriminator')
    parser.add_argument('--g_steps', type=int, default=1,
                    help='number of steps updating generator')
    parser.add_argument('--d_steps', type=int, default=1,
                    help='number of steps updating discriminator')
    parser.add_argument('--d_gen_weight', type=float, default=0.5,
                    help='weight on generated sent loss for discriminator')
    parser.add_argument('--d_mm_weight', type=float, default=0.5,
                    help='weight on mismatched sent loss for discriminator')
    parser.add_argument('--d_bidirectional', type=int, default=1,
                    help='use bidirectional lstm for discriminator')
    parser.add_argument('--noise', type=int, default=0,
                    help='use noise for generator (1 = yes, 0 = no)')
    parser.add_argument('--noise_size', type=int, default=100,
                    help='size of noise')
    # parser.add_argument('--gan', type=int, default=1,
    #                     help='train with gan (1 = yes, 0 = no)?')


    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"
    assert args.save_checkpoint_every >= 0, "saving checkpoint at every $epoch should be non-negative"

    return args
