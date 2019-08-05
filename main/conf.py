import argparse
'''
    define the basic configuration parameters,
    also define one command-lines argument parsing method: init_args
'''

MAX_SEQUENCE = 30        # 最大的识别汉字的长度
MASK_VALUE = 0
CHARSET = "data/charset.txt" # 3770的一级字库
INPUT_IMAGE_HEIGHT = 32  # 图像归一化的高度
INPUT_IMAGE_WIDTH = 200  # 最大的图像宽度
GRU_HIDDEN_SIZE = 64     # GRU隐含层神经元数量

# 伐喜欢tensorflow的flags方式，使用朴素的argparse
# dislike the flags style of tensorflow, instead using argparse
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs" ,default=1,type=int,help="")
    parser.add_argument("--batch" , default=1,type=int,help="")
    parser.add_argument("--workers",default=1,type=int,help="")
    parser.add_argument("--validation_steps",default=1,type=int, help="")
    parser.add_argument("--validation_batch",default=1,type=int, help="")
    parser.add_argument("--early_stop", default=1, type=int, help="")
    args = parser.parse_args()
    return args
