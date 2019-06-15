from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

from six.moves import cPickle

def zero_pad(features,n_feat):
    if features.shape[0] < n_feat:
        features = np.vstack((features,np.zeros((n_feat - features.shape[0], features.shape[1]))))
    return features

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

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_wtoi(self):
        return self.word_to_ix

    def get_seq_length(self):
        return self.seq_length

    def build_activity_dict(self):
        self.activity_dict = {}
        videos = self.info['videos']
        for ix in range(len(self.video_id)):
            v_ix = self.video_id[ix]
            for act in videos[v_ix]['activities']:
                  if act not in self.activity_dict:
                      self.activity_dict[act] = [ix]
                  else:
                      self.activity_dict[act].append(ix)

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.input_fc_dir = self.opt.input_fc_dir

        # use other features
        self.use_img = getattr(opt, 'use_img', 0) or getattr(opt, 'd_use_img', 0)
        if self.use_img:
            self.input_img_dir = self.opt.input_img_dir
        self.use_box = getattr(opt, 'use_box', 0) or getattr(opt, 'd_use_box', 0)
        if self.use_box:
            self.input_box_dir = self.opt.input_box_dir
        self.feat_type = opt.feat_type
        self.nbox = 3

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.word_to_ix = self.info['word_to_ix']
        self.vocab_size = len(self.ix_to_word)
        self.ix_to_activity = self.info.get('ix_to_activity',None)
        self.activity_size = len(self.ix_to_activity) if self.ix_to_activity is not None else 0
        print('vocab size is ', self.vocab_size)
        print('activity size is', self.activity_size)

        # open the hdf5 file containing visual features and captions
        print('DataLoader loading h5 file: ', opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r')
        self.labels = self.h5_label_file['labels'].value
        seq_size = self.labels.shape
        self.seq_length = seq_size[2]
        self.max_sent_num = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        self.max_seg = opt.max_seg
        self.sent_num = self.h5_label_file['sent_num'].value
        self.video_id = self.h5_label_file['video_id'].value
        self.timestamp = self.h5_label_file['timestamp'].value
        if self.activity_size > 0:
            self.activity = self.h5_label_file['activity'].value
        print('number of captions is', sum(self.sent_num))
        self.mean = opt.use_mean
        self.negatives = opt.negatives

        self.build_activity_dict()

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        self.split_size = {'train': 0, 'val': 0, 'test': 0}
        self.ix_split = {}
        for j in range(seq_size[0]):
            i = self.video_id[j]
            video = self.info['videos'][i]
            if video['split'] == 'train':
                self.split_ix['train'].append(j)
                self.split_size['train']+=1
                self.ix_split[j] = 'train'
            elif video['split'] == 'val_2':
                self.split_ix['val'].append(j)
                self.split_size['val']+=1
                self.ix_split[j] = 'val'
            elif video['split'] == 'val_1':
                self.split_ix['test'].append(j)
                self.split_size['test']+=1
                self.ix_split[j] = 'test'
            elif opt.train_only: # restval
                self.split_ix['train'].append(j)
                self.split_size['train']+=1

        print('assigned %d videos to split train' % len(self.split_ix['train']))
        print('assigned %d videos to split val' % len(self.split_ix['val']))
        print('assigned %d videos to split test' % len(self.split_ix['test']))


        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)


    # mean pool the features across $max_seg segments
    def meanpool_segments(self, features):
        if features.shape[0] >= self.max_seg:
            tmp_feat = []
            nps = int(np.floor(features.shape[0] // self.max_seg))  # numbers per segment
            for i in range(self.max_seg):
                if i != self.max_seg - 1:
                    segment = features[nps * i:nps * (i + 1)]
                else:
                    segment = features[nps * i:]
                segment = segment.mean(axis=0)
                tmp_feat.append(segment)
            features = np.array(tmp_feat)
        else:
            # 0 pad frames
            features = zero_pad(features, self.max_seg)
        return features

    def get_seg_batch(self, index, mode):
        v_idx = self.video_id[index]
        id = self.info['videos'][v_idx]['id']
        duration = self.info['videos'][v_idx]['duration']
        timestamp = self.timestamp[index]
        sent_num = self.sent_num[index]
        assert sent_num > 0, 'data should have at least one caption'
        features = []
        if mode == 'video':
            tmp_fc_all = np.load(os.path.join(self.input_fc_dir, id + '.npy'))
        elif mode == 'img':
            if not self.use_img:
                return None
            tmp_fc_all = np.load(os.path.join(self.input_img_dir, id + '.npy'))
        else:
            raise AttributeError("mode %s not found" % mode)
        for i in range(sent_num):
            weight = tmp_fc_all.shape[0] / duration
            start = int(timestamp[i, 0] * weight)
            end = int(timestamp[i, 1] * weight) + 1
            assert start < end, 'end index: %d should be greater than start index: %d ' % (end, start)
            tmp_fc = tmp_fc_all[start:end]
            features.append(self.meanpool_segments(tmp_fc))
        return np.array(features)

    def get_box_batch(self, index):
        if not self.use_box:
            return None
        v_idx = self.video_id[index]
        id = self.info['videos'][v_idx]['id']
        sent_num = self.sent_num[index]
        assert sent_num > 0, 'data should have at least one caption'
        box_features = []
        split = self.ix_split[index]
        if split == 'val':
            split = 'val2'
        elif split == 'test':
            split = 'val1'
        dir = os.path.join(self.input_box_dir,split)
        feats = np.load(os.path.join(dir,id + '.npy'))
        assert feats.shape[0] >= 3 * sent_num, 'weird feature for %s' % id
        for i in range(sent_num):
            box_features.append(feats[i*self.nbox:(i+1)*self.nbox])
        return box_features

    def set_negatives(self,mode):
        self.negatives = mode

    # Each batch is a video with multiple clips/sentences
    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        # inputs for training
        label_batch = np.zeros((batch_size, self.max_sent_num, self.seq_length + 2), dtype = 'int')
        mask_batch = np.zeros((batch_size, self.max_sent_num, self.seq_length + 2), dtype='float32')
        sent_num_batch = np.zeros(batch_size, dtype='int')
        fc_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.fc_feat_size], dtype = 'float32')
        img_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.img_feat_size], dtype = 'float32')
        box_batch = np.zeros([batch_size, self.max_sent_num, self.nbox, self.opt.box_feat_size], dtype = 'float32')

        # negative inputs for discriminator
        mm_fc_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.fc_feat_size], dtype = 'float32')
        mm_img_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.opt.img_feat_size], dtype = 'float32')
        mm_box_batch = np.zeros([batch_size, self.max_sent_num, self.nbox, self.opt.box_feat_size], dtype = 'float32')
        act_batch = []
        mm_act_batch = []
        mm_batch = np.zeros((batch_size, self.max_sent_num, self.seq_length + 2), dtype = 'int')
        shuffle_batch = np.zeros([batch_size, self.max_sent_num, self.max_seg, self.seq_length+2], dtype = 'float32')

        wrapped = False
        infos = []
        gts = []
        for i in range(batch_size):
            # fetch visual features
            tmp_fcs, ix, tmp_wrapped = self._prefetch_process[split].get()
            sent_num = self.sent_num[ix]
            fc_batch[i,:sent_num] = tmp_fcs[0]
            img_batch[i,:sent_num] = tmp_fcs[1]
            box_batch[i,:sent_num] = tmp_fcs[2]
            sent_num_batch[i] = sent_num
            label_batch[i, :, 1 : self.seq_length + 1] = self.labels[ix]
            v_ix = self.video_id[ix]

            # get visually mismatched (mm) captions and features as inputs to generator and visual discriminator
            if self.negatives == 'hard':  # get caption from video with same activity (hard negatives)
                activity = self.info['videos'][v_ix]['activities'][i % 2 - 1]  # randomly choose first or last activity
                m = 0
                while True:
                    a_ix = random.choice(self.activity_dict[activity])
                    n = min(sent_num - m, self.sent_num[a_ix])
                    if self.video_id[a_ix] != v_ix:  # avoid getting the gt pair
                        mm_batch[i, m:m + n, 1:self.seq_length + 1] = self.labels[a_ix, :n, :]
                        mm_fc_batch[i, m:m + n] = self.get_seg_batch(a_ix, "video")[:n]
                        mm_img_batch[i, m:m + n] = self.get_seg_batch(a_ix, "img")[:n] if self.use_img else None
                        mm_box_batch[i, m:m + n] = self.get_box_batch(a_ix)[:n] if self.use_box else None
                        m += n
                    if m >= sent_num:
                        break
            else:  # get random caption (random negatives)
                while True:
                    mmix = random.randint(0, len(self.split_ix[split]) - 1)
                    if self.video_id[mmix] != v_ix and sent_num <= self.sent_num[mmix]:  # avoid getting the gt pair
                        mm_batch[i, :sent_num, 1:self.seq_length + 1] = self.labels[mmix, :sent_num, :]
                        mm_fc_batch[i, :sent_num] = self.get_seg_batch(mmix, "video")[:sent_num]
                        mm_img_batch[i, :sent_num] = self.get_seg_batch(mmix, "img")[
                                                     :sent_num] if self.use_img else None
                        mm_box_batch[i, :sent_num] = self.get_box_batch(mmix)[:sent_num] if self.use_box else None
                        break

            if tmp_wrapped:
                wrapped = True

            gts.append(self.labels[ix])
            for j in range(sent_num):
                info_dict = {}
                info_dict['index'] = ix
                info_dict['id'] = self.info['videos'][v_ix]['id']
                info_dict['file_path'] = self.info['videos'][v_ix]['path']
                info_dict['timestamp'] = self.timestamp[ix][j]
                info_dict['activity'] = self.info['videos'][v_ix]['activities']
                infos.append(info_dict)

            # generate mask
            nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, label_batch[i])))
            for ix, row in enumerate(mask_batch[i]):
                if ix < sent_num:
                    row[:nonzeros[ix]] = 1

        data = {}

        data['fc_feats'] = np.array(fc_batch)
        data['img_feats'] = np.array(img_batch)
        data['box_feats'] = np.array(box_batch)
        data['labels'] = np.array(label_batch)
        data['sent_num'] = np.array(sent_num_batch)

        data['mm_fc_feats'] = np.array(mm_fc_batch)
        data['mm_img_feats'] = np.array(mm_img_batch)
        data['mm_box_feats'] = np.array(mm_box_batch)
        data['mm_labels'] = np.array(mm_batch)
        data['shufflelabels'] = np.array(shuffle_batch)

        data['activities'] = np.array(act_batch,dtype='int') if self.activity_size > 0 else None
        data['mm_activities'] = np.array(mm_act_batch,dtype='int') if self.activity_size > 0 else None

        data['masks'] = mask_batch
        data['gts'] = np.array(gts)
        data['att_feats'] = None
        data['att_masks'] = None
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        return [self.get_seg_batch(index,"video"), self.get_seg_batch(index,"img"),
                self.get_box_batch(index)], index

    def __len__(self):
        return len(self.info['videos'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        assert tmp[1] == ix, "ix not equal"

        return tmp + [wrapped]