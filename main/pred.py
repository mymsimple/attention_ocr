from utils import util, logger as log,label_utils
from main import conf
from layers import model as _model
import logging,cv2
import numpy as np

logger = logging.getLogger("Train")

def pred(args):
    charset = label_utils.get_charset(conf.CHARSET)
    conf.CHARSET_SIZE = len(charset)

    # 定义模型
    _,encoder_model,decoder_model = _model.inference_model(conf,args)

    # 分别加载模型
    encoder_model.load_model(args.model)
    decoder_model.load_model(args.model)
    logger.info("加载了模型：%s", args.model)

    logger.info("开始预测图片：%s",args.image)
    image = cv2.imread(args.image)
    test_fr_seq = sents2sequences(fr_tokenizer, ['sos'], fr_vsize)

    test_en_onehot_seq = to_categorical(test_en_seq, num_classes=en_vsize)
    test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

    # 编码器先预测
    encoder_outs, encoder_fwd_state, encoder_back_state = encoder_model.predict(image)

    # 准备编码器的初始输入状态
    decoder_init_state = np.concatenate([encoder_fwd_state, encoder_back_state], axis=-1)

    attention_weights = []
    fr_text = ''

    # 开始预测字符
    for i in range(conf.MAX_SEQUENCE):

        decoder_inputs = conf.STX
        encoder_out_states =
        decoder_init_state =

        # infer_decoder_model : Model(inputs=[decoder_inputs, encoder_out_states,decoder_init_state], outputs=[decoder_pred,attn_states,decoder_state])
        decoder_out, attention, decoder_init_state = decoder_model.predict([decoder_inputs, encoder_out_states,decoder_init_stat])
        decoder_index = np.argmax(decoder_out, axis=-1)[0, 0]

        if decoder_index == 2:
            logger.info("预测字符为ETX，退出")
            break #==>conf.CHAR_ETX: break

        to_categorical()

        test_fr_seq = sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], fr_vsize)
        test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

        attention_weights.append((dec_ind, attention))
        fr_text += fr_index2word[dec_ind] + ' '

    return fr_text, attention_weights

def sents2sequences(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):
    encoded_text = tokenizer.texts_to_sequences(sentences)
    preproc_text = pad_sequences(encoded_text, padding=padding_type, maxlen=pad_length, value=0)
    if reverse:
        preproc_text = np.flip(preproc_text, axis=1)

    return preproc_text


if __name__ == "__main__":
    log.init()
    args = conf.init_pred_args()
    pred(args)