from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .VideoDiscriminator5 import VideoDiscriminator5
from .HybridDiscriminator import HybridDiscriminator
from .MultiModalGenerator import MultiModalGenerator
# from .ShowTellModel import ShowTellModel
# from .FCModel import FCModel
# from .OldModel import ShowAttendTellModel, AllImgModel
# from .AttModel import *

def setup(opt):

    model = MultiModalGenerator(opt)
    # elif opt.caption_model == 'fc':
    #     model = FCModel(opt)
    # elif opt.caption_model == 'show_tell':
    #     model = ShowTellModel(opt)
    # # Att2in model in self-critical
    # elif opt.caption_model == 'att2in':
    #     model = Att2inModel(opt)
    # # Att2in model with two-layer MLP img embedding and word embedding
    # elif opt.caption_model == 'att2in2':
    #     model = Att2in2Model(opt)
    # elif opt.caption_model == 'att2all2':
    #     model = Att2all2Model(opt)
    # # Adaptive Attention model from Knowing when to look
    # elif opt.caption_model == 'adaatt':
    #     model = AdaAttModel(opt)
    # # Adaptive Attention with maxout lstm
    # elif opt.caption_model == 'adaattmo':
    #     model = AdaAttMOModel(opt)
    # # Top-down attention model
    # elif opt.caption_model == 'topdown':
    #     model = TopDownModel(opt)
    # # StackAtt
    # elif opt.caption_model == 'stackatt':
    #     model = StackAttModel(opt)
    # # DenseAtt
    # elif opt.caption_model == 'denseatt':
    #     model = DenseAttModel(opt)
    # else:
    #     raise Exception("Caption model not supported: {}".format(opt.caption_model))

    dis_model = HybridDiscriminator(opt)

    return model,dis_model
