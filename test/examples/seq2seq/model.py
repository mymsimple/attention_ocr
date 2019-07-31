#from keras.layers import LSTM,Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.python.keras.models import Model
import tensorflow as tf

def _p(t, name):
    return tf.Print(t, [tf.shape(t)], name + "\n")

def model(hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
    print("hidden_size:%r, batch_size:%r, en_timesteps:%r, en_vsize:%r, fr_timesteps:%r, fr_vsize:%r" % (hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize))

    encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
    decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')

    # 编码器
    encoder_bi_gru = Bidirectional(GRU(units=hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       name='encoder_gru'),
                                   name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_bi_gru(encoder_inputs)


    # 解码器
    decoder_gru = GRU(units=hidden_size * 2,
                      return_sequences=True,
                      return_state=True,
                      name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(
        inputs=decoder_inputs,
        initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state]))

    # Dense layer
    dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_out)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary()

    return full_model