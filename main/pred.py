from utils import util, logger as log,label_utils
from main import conf
import logging,cv2

logger = logging.getLogger("Train")


def pred(args):
    # TF调试代码 for tf debugging：
    # from tensorflow.python import debug as tf_debug
    # from tensorflow.python.keras import backend as K
    # sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    charset = label_utils.get_charset(conf.CHARSET)
    conf.CHARSET_SIZE = len(charset)
    # model = _model.model(conf)

    from keras.models import load_model
    model = load_model(args.model)#, custom_objects={'AttentionLayer': AttentionLayer})
    logger.info("加载了模型：%s", args.model)
    logger.info("开始预测图片：%s",args.image)

    image = cv2.imread(args.image)
    result = model.predict(image,verbose=True)


if __name__ == "__main__":
    log.init()
    args = conf.init_pred_args()
    pred(args)