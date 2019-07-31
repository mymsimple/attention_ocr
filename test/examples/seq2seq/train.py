import tensorflow.keras as keras
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os, sys

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
if project_path not in sys.path:
    sys.path.append(project_path)

from test.examples.utils.data_helper import read_data, sents2sequences
from test.examples.seq2seq import model
from test.examples.utils.logger import get_logger

base_dir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
logger = get_logger("examples.nmt_bidirectional.train", os.path.join(base_dir, 'logs'))

batch_size = 1
hidden_size = 10
en_timesteps, fr_timesteps = 5, 15


def get_data(train_size, random_seed=100):

    """ Getting randomly shuffled training / testing data """
    en_text = read_data(os.path.join(project_path, '../data', 'small_vocab_en.txt'))
    fr_text = read_data(os.path.join(project_path, '../data', 'small_vocab_fr.txt'))
    logger.info('Length of text: {}'.format(len(en_text)))

    fr_text = ['sos ' + sent[:-1] + 'eos' if sent.endswith('.') else 'sos ' + sent + ' eos' for sent in fr_text]

    np.random.seed(random_seed)
    inds = np.arange(len(en_text))
    np.random.shuffle(inds)

    train_inds = inds[:train_size]
    test_inds = inds[train_size:]
    tr_en_text = [en_text[ti] for ti in train_inds]
    tr_fr_text = [fr_text[ti] for ti in train_inds]

    ts_en_text = [en_text[ti] for ti in test_inds]
    ts_fr_text = [fr_text[ti] for ti in test_inds]

    logger.info("Average length of an English sentence: {}".format(
        np.mean([len(en_sent.split(" ")) for en_sent in tr_en_text])))
    logger.info("Average length of a French sentence: {}".format(
        np.mean([len(fr_sent.split(" ")) for fr_sent in tr_fr_text])))
    return tr_en_text, tr_fr_text, ts_en_text, ts_fr_text

def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
    """ Preprocessing data and getting a sequence of word indices """

    en_seq = sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
    fr_seq = sents2sequences(fr_tokenizer, fr_text, pad_length=fr_timesteps)
    logger.info('Vocabulary size (English): {}'.format(np.max(en_seq)+1))
    logger.info('Vocabulary size (French): {}'.format(np.max(fr_seq)+1))
    logger.debug('En text shape: {}'.format(en_seq.shape))
    logger.debug('Fr text shape: {}'.format(fr_seq.shape))

    return en_seq, fr_seq


def train(full_model, en_seq, fr_seq, batch_size, n_epochs=10):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, en_seq.shape[0] - batch_size, batch_size):
            en_onehot_seq = to_categorical(en_seq[bi:bi + batch_size, :], num_classes=en_vsize)
            fr_onehot_seq = to_categorical(fr_seq[bi:bi + batch_size, :], num_classes=fr_vsize)

            full_model.train_on_batch([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :])

            l = full_model.evaluate([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :],
                                    batch_size=batch_size, verbose=0)
            losses.append(l)
            logger.info("Loss in {}/{}: {}".format(bi,ep + 1, np.mean(losses)))


if __name__ == '__main__':
    debug = False

    train_size = 100
    filename = ''
    tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data(train_size=train_size)

    # print(tr_en_text, tr_fr_text)
    """ Defining tokenizers """
    en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    en_tokenizer.fit_on_texts(tr_en_text)

    fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    fr_tokenizer.fit_on_texts(tr_fr_text)

    """ Getting preprocessed data """
    en_seq, fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, tr_en_text, tr_fr_text, en_timesteps, fr_timesteps)

    en_vsize = len(en_tokenizer.word_index.keys())
    fr_vsize = len(fr_tokenizer.word_index.keys())

    """ Defining the full model """
    full_model = model.model(
        hidden_size=hidden_size,
        batch_size=batch_size,
        en_timesteps=en_timesteps,
        fr_timesteps=fr_timesteps,
        en_vsize=en_vsize,
        fr_vsize=fr_vsize)

    n_epochs = 1
    train(full_model, en_seq, fr_seq, batch_size, n_epochs)