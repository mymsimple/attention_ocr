from layers import model as _model
from utils.sequence import SequenceData
from utils import util, logger as log,label_utils
from tensorflow.python.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint
from main import conf
import logging

logger = logging.getLogger("Train")


def train(args):
    # TF调试代码 for tf debugging：
    # from tensorflow.python import debug as tf_debug
    # from tensorflow.python.keras import backend as K
    # sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    charset = label_utils.get_charset(conf.CHARSET)
    conf.CHARSET_SIZE = len(charset)
    model = _model.model(conf)

    train_sequence = SequenceData(name="训练",
                                  label_file="data/train.txt",
                                  charset_file="data/charset.txt",
                                  conf=conf,
                                  batch_size=args.batch)
    valid_sequence = SequenceData(name="验证",
                                  label_file="data/validate.txt",
                                  charset_file="data/charset.txt",
                                  conf=conf,
                                  batch_size=args.validation_batch)

    timestamp = util.timestamp_s()
    tb_log_name = "logs/tboard/{}".format(timestamp)
    checkpoint_path = "model/checkpoint/checkpoint-{}.hdf5".format(timestamp)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max')

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=args.early_stop,
        verbose=1,
        mode='max')

    logger.info("Begin train开始训练：")

    model.fit_generator(
        generator=train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=args.epochs,
        workers=args.workers,
        callbacks=[TensorBoard(log_dir=tb_log_name),checkpoint,early_stop],
        use_multiprocessing=True,
        validation_data=valid_sequence,
        validation_steps=1)

    logger.info("Train end训练结束!")

    model_path = "model/ocr-attention-{}.hdf5".format(util.timestamp_s())
    model.save(model_path)
    logger.info("Save model保存训练后的模型到：%s", model_path)


if __name__ == "__main__":
    log.init()
    args = conf.init_args()
    train(args)