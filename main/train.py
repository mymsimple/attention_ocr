from main import logger
from main import full_model
from main.sequence import SequenceData
from utils import util
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

def train():

    model = full_model.model()

    train_sequence = SequenceData("data/train.txt","data/charset.txt")
    valid_sequence = SequenceData("data/train.txt","data/charset.txt")

    timestamp = util.timestamp_s()
    tb_log_name = "logs/tboard/{}".format(timestamp)
    checkpoint_path = "model/checkpoint/checkpoint-{}.hdf5".format(timestamp)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    logger.info("开始训练：")

    model.fit_generator(generator=train_sequence,
        # steps_per_epoch=int(len(D)),
        epochs=2,
        workers=3,
        callbacks=[TensorBoard(log_dir=tb_log_name),checkpoint],
        use_multiprocessing=True,
        validation_data=valid_sequence,
        validation_steps=1)

    logger.info("训练结束!")

    model_path = "model/ocr-attention-{}.hdf5".format(util.timestamp_s())
    model.save(model)
    logger.info("保存训练后的模型到：%s", model_path)

if __name__ == "__main__":
    logger.init()
    train()