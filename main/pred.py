from utils import util, logger as log,label_utils
from main import conf
from layers import model as _model
from layers.conv import Conv
import logging,cv2
import numpy as np
from layers.attention import AttentionLayer
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import load_model

logger = logging.getLogger(__name__)

def pred(args):
    charset = label_utils.get_charset(conf.CHARSET)
    CHARSET_SIZE = len(charset)

    # 加载模型
    model = load_model(args.model, custom_objects={
        'words_accuracy': _model.words_accuracy,
        'squeeze_wrapper': Conv().squeeze_wrapper,
        'AttentionLayer':AttentionLayer})
    encoder_model,decoder_model = _model.infer_model(model,conf)
    logger.info("加载了模型：%s", args.model)

    logger.info("开始预测图片：%s",args.image)
    image = cv2.imread(args.image)

    # 编码器先预测
    encoder_out_states, encoder_state = encoder_model.predict(np.array([image]))

    # 准备编码器的初始输入状态
    attention_weights = []

    # 开始是STX
    from utils.label_utils import convert_to_id
    decoder_index = convert_to_id([conf.CHAR_STX], charset)
    decoder_state = encoder_state

    result = ""

    # 解码器解码，开始一个一个地预测字符
    for i in range(conf.MAX_SEQUENCE):

        # 别看又padding啥的，其实就是一个字符，这样做是为了凑输入的维度定义
        decoder_inputs = to_categorical(decoder_index,num_classes=CHARSET_SIZE)

        # 只解码一个字符,decoder_state被更新
        decoder_out, attention,decoder_state = decoder_model.predict([[decoder_inputs],encoder_out_states,decoder_state])

        # decoder_out[1,1,3770] =argmax=> [[max_id]]
        decoder_index = decoder_out.argmax(axis=2)
        decoder_index = decoder_index[0]
        pred_char = label_utils.id2str(decoder_index, charset)
        if pred_char == conf.CHAR_ETX:
            logger.info("预测字符为ETX，退出")
            break #==>conf.CHAR_ETX: break

        attention_weights.append(attention)

        logger.info("预测字符ID[%d],对应字符[%s]",decoder_index[0],pred_char)
        result+= pred_char
        decoder_index = [decoder_index]

    if len(result)>=conf.MAX_SEQUENCE:
        logger.debug("预测字符为：%s，达到最大预测长度", result)
    else:
        logger.debug("预测字符为：%s，解码最后为ETX", result)

    return pred_char,attention_weights


if __name__ == "__main__":
    log.init()
    charset = label_utils.get_charset(conf.CHARSET)
    conf.CHARSET_SIZE = len(charset)
    args = conf.init_pred_args()
    result,attention_probs = pred(args)
    logger.info("预测字符串为：%s",result)
    # logger.info("注意力概率为：%r", attention_probs)