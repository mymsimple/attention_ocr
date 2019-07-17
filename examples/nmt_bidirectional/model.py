from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.python.keras.models import Model
from layers.attention import AttentionLayer
import tensorflow as tf

def _p(t,name):
    return tf.Print(t,[tf.shape(t)],name+"\n")

#
def define_nmt(hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
    """ Defining a NMT model """

    # Define an input sequence and process it.
    if batch_size:
        encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')
    else:
        encoder_inputs = Input(shape=(en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(shape=(fr_timesteps - 1, fr_vsize), name='decoder_inputs')

    # Encoder GRU
    encoder_gru = Bidirectional(GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru'), name='bidirectional_encoder')
    # encoder_inputs = _p(encoder_inputs,"encoder_inputs")
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_gru(encoder_inputs)
    # encoder_out = _p(encoder_out,"encoder_out")

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(hidden_size*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(
        decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state])
    )
    # decoder_inputs = _p(decoder_inputs,"decoder_inputs")
    # decoder_out = _p(decoder_out,"decoder_out")

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])
    # attn_out = _p(attn_out,"attn_out")

    # Concat attention input and decoder GRU output
    print("decoder_out:",tf.shape("decoder_out"))
    print("attn_out:", tf.shape("attn_out"))
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')

    # print("decoder_concat_input:", tf.shape("decoder_concat_input"))
    # decoder_pred = dense_time(decoder_concat_input) # ???
    # print("decoder_concat_input:", tf.shape("decoder_concat_input"))

    decoder_pred = dense_time(decoder_concat_input) # ???
    # decoder_pred = _p(decoder_pred,"decoder_pred")
    
    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary()

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
    encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, 2*hidden_size), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(batch_size, 2*hidden_size), name='decoder_init')

    decoder_inf_out, decoder_inf_state = decoder_gru(
        decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

    return full_model, encoder_model, decoder_model


if __name__ == '__main__':

    """ Checking nmt model for toy examples """
    define_nmt(64, None, 20, 30, 20, 20)