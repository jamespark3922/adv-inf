# Adversarial Inference for Multi-Sentence Video Descriptions

This is the implementation of [Adversarial Inference for Multi-Sentence Video Descriptions](https://arxiv.org/pdf/1812.05634.pdf)

This repository is based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). Thank you Ruotian for the code! The modifications are:
- Training Multimodal Generator and Hybrid Discriminator in `models/`.
- Adversarial Inference in `eval_utils.py`

## Requirements
Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)  
PyTorch 0.4 (along with torchvision)  
[densevid_eval](https://github.com/jamespark3922/densevid_eval) (for activitynet evaluation)  
java to run meteor.jar file

## Training on ActivityNet Dense Captions

### Download ActivityNet captions and preprocess them
We share the input labels and features in this [folder](https://drive.google.com/drive/u/0/folders/1Xaw8yaVa-V63KOL3m3JnRFfkmPtgfeO1). (Scripts to preprocess the labels will be available soon.)

### Features
- **renext101-64f (126GB)** extracted from [r3d repository](https://github.com/kenshohara/video-classification-3d-cnn-pytorch), mean-pooled into 10 segments for eah clip
- **resnet152 (14GB)**, extracted 100 frames for each video
- **bottomup labels (16GB)** with confidence score, extracted 3 frames for each clip

After downloading them all, unzip them to your preferred feature directory.

Note that mean-pooling operations are done when loading the data in `dataloader.py`

### Training
```bash
python train.py --caption_model video --input_json activity_net/inputs/video_data_dense.json --input_fc_dir activity_net/feats/resnext101-64f/ --input_img_dir activity_net/feats/resnet152/ --input_box_dir activity_net/feats/bottomup/ --input_label_h5 activity_net/inputs/video_data_dense_label.h5 --glove_npy activity_net/inputs/glove.npy --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path video_ckpt --val_videos_use -1 --losses_print_every 10 --batch_size 16 --language_eval 1
```
**Context**: The generator model uses the hidden state of previous sentence as "context", starting at epoch `--g_context_epoch`.

### Evaluation
After training is done, evaluate the captions in paragraph level. Note the evaluation is done on **val1** set.

The normal inference using greedymax or beamsearch can be run with the following command:
```angular2html
python eval.py --g_model_path video_ckpt/gen_best.pth --infos_path video_ckpt/infos.pkl --d_model_path video_ckpt/dis_best.pth --sample_max 1 --id $id --beam_size $beam_size
```
and will be saved in `densevid_eval/caption_$id.json`

You can disable `--d_model_path` if you do not wish to score and evaluate the discriminator.

Adversarial Inference (sampling $num_samples sentences and choosing the best one with discriminator) can be run with 
```angular2html
python eval.py --g_model_path video_ckpt/gen_best.pth --infos_path video_ckpt/infos.pkl --d_model_path video_ckpt/dis_best.pth --sample_max 0 --num_samples $num_samples --temperature $temperature --id $id
```

You can also run the diversity metrics **(Div-N, Re-N)** in paper.
```angular2html
python evaluateCaptionsDiversity.py -s $submission_file
```

## Reference

```
@article{park2018advinf,
  title= Adversarial Inference for Multi-Sentence Video Descriptions,
  author={Park, Jae Sung and Rohrbach, Marcus and Darrell, Trevor and Rohrbach, Anna},
  jorunal={CVPR 2019},
  year={2018}
}

@article{luo2018discriminability,
  title={Discriminability objective for training descriptive captions},
  author={Luo, Ruotian and Price, Brian and Cohen, Scott and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1803.04376},
  year={2018}
}
```