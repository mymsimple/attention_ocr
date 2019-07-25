import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

def _p(t,name):
    return tf.Print(t,[tf.shape(t)],name)


def one_step_of_attention(h_prev, a):
    """
    Get the context.

    Input:
    h_prev - Previous hidden state of a RNN layer (m, n_h)
    a - Input data, possibly processed (m, Tx, n_a)

    Output:
    context - Current context (m, Tx, n_a)
    """
    # Repeat vector to match a's dimensions
    h_repeat = at_repeat(h_prev)
    # Calculate attention weights
    i = at_concatenate([a, h_repeat])
    i = at_dense1(i)
    i = at_dense2(i)
    attention = at_softmax(i)
    # Calculate the context
    context = at_dot([attention, a])

    return context



# states,我理解就是S_t-1
def energy_step(S_t_1,): # inputs(batch,dim)
    inputs = _p(S_t_1 ,"energy_step:S_t_1 算能量函数了..........")  # S_t_1：[1，20]


    en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
    de_hidden = S_t_1.shape[-1]

    #  W * h_j
    reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
    W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))

    # U * S_t - 1
    U_a_dot_h = K.expand_dims(K.dot(, self.U_a), 1)  # <= batch_size, 1, latent_dim

    # tanh ( W * h_j + U * S_t-1 + b )
    reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))

    # V * tanh ( W * h_j + U * S_t-1 + b )
    e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))

    # softmax(e_tj)
    e_i = K.softmax(e_i)
    e_i = _p(e_i ,"energy_step:e_i")
    return e_i, [e_i]


def attention_layer(X, n_h, Ty):
    """
    Creates an attention layer.

    Input:
    X - Layer input (m, Tx, x_vocab_size)
    n_h - Size of LSTM hidden layer
    Ty - Timesteps in output sequence

    Output:
    output - The output of the attention layer (m, Tx, n_h)
    """
    # Define the default state for the LSTM layer
    h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)), name='h_attention_layer')(X)
    c = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)), name='c_attention_layer')(X)
    # Messy, but the alternative is using more Input()

    at_LSTM = LSTM(n_h, return_state=True, name='at_LSTM_attention_layer')

    output = []

    # Run attention step and RNN for each output time step
    for _ in range(Ty):
        context = one_step_of_attention(h, X)

        h, _, c = at_LSTM(context, initial_state=[h, c])

        output.append(h)

    return output


def get_model(Tx, Ty, layer1_size, layer2_size, x_vocab_size, y_vocab_size):
    """
    Creates a model.

    input:
    Tx - Number of x timesteps
    Ty - Number of y timesteps
    size_layer1 - Number of neurons in BiLSTM
    size_layer2 - Number of neurons in attention LSTM hidden layer
    x_vocab_size - Number of possible token types for x
    y_vocab_size - Number of possible token types for y

    Output:
    model - A Keras Model.
    """

    # Create layers one by one
    X = Input(shape=(Tx, x_vocab_size), name='X_Input')

    a1 = Bidirectional(LSTM(layer1_size, return_sequences=True), merge_mode='concat', name='Bid_LSTM')(X)

    a2 = attention_layer(a1, layer2_size, Ty)

    a3 = [layer3(timestep) for timestep in a2]

    # Create Keras model
    model = Model(S_t_1=[X], outputs=a3)

    return model
