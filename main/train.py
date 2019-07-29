from main import logger as log
from main import full_model
from main.sequence import SequenceData
from utils import util
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence

import logging

logger = logging.getLogger("Train")

def train():

    model = full_model.model()

    train_sequence = SequenceData("训练","data/train.txt","data/charset.txt",2)
    valid_sequence = SequenceData("验证","data/validate.txt","data/charset.txt",2)
    # print(isinstance(valid_sequence, Sequence))

    timestamp = util.timestamp_s()
    tb_log_name = "logs/tboard/{}".format(timestamp)
    checkpoint_path = "model/checkpoint/checkpoint-{}.hdf5".format(timestamp)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    logger.info("开始训练：")

    model.fit_generator(generator=train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=3,
        workers=1,
        callbacks=[TensorBoard(log_dir=tb_log_name),checkpoint],
        use_multiprocessing=True,
        validation_data=valid_sequence,
        validation_steps=1)

    logger.info("训练结束!")

    model_path = "model/ocr-attention-{}.hdf5".format(util.timestamp_s())
    model.save(model)
    logger.info("保存训练后的模型到：%s", model_path)

if __name__ == "__main__":
    log.init()
    train()