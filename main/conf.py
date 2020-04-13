import argparse
'''
    define the basic configuration parameters,
    also define one command-lines argument parsing method: init_args
'''

CHAR_ETX = "\x03"
CHAR_STX = "\x02"
CHAR_NULL = " "

MAX_SEQUENCE = 30        # 最大的识别汉字的长度
MASK_VALUE = 0
CHARSET = "data/charset.txt" # 3770的一级字库
INPUT_IMAGE_HEIGHT = 32  # 图像归一化的高度
INPUT_IMAGE_WIDTH = 256  # 最大的图像宽度
GRU_HIDDEN_SIZE = 256     # GRU隐含层神经元数量

DIR_LOGS="logs"
DIR_TBOARD="logs/tboard"
DIR_MODEL="model"
DIR_CHECKPOINT="model/checkpoint"

# 伐喜欢tensorflow的flags方式，使用朴素的argparse
# dislike the flags style of tensorflow, instead using argparse
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name" ,default="attention_ocr",type=str,help="")
    parser.add_argument("--train_label_file",    default="data/train.txt",    type=str, help="")
    parser.add_argument("--validate_label_file", default="data/validate.txt", type=str, help="")
    parser.add_argument("--epochs" ,default=1,type=int,help="")
    parser.add_argument("--debug_step", default=1,type=int,help="")
    parser.add_argument("--steps_per_epoch", default=None,type=int,help="")
    parser.add_argument("--batch" , default=1,type=int,help="")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="")
    parser.add_argument("--workers",default=1,type=int,help="")
    parser.add_argument("--retrain", default=False, type=bool, help="")
    parser.add_argument("--preprocess_num",default=1,type=int,help="")
    parser.add_argument("--validation_steps",default=1,type=int, help="")
    parser.add_argument("--validation_batch",default=1,type=int, help="")
    parser.add_argument("--early_stop", default=1, type=int, help="")
    args = parser.parse_args()
    print("==============================")
    print("所有的使用的配置：")
    print("==============================")
    print(args)
    return args


def init_pred_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image" ,default=1,type=str,help="")
    parser.add_argument("--model" ,default=1,type=str,help="")
    args = parser.parse_args()
    return args
