# notes


按照正确的答案输入到decoder

```python
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')


embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)

del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

# =============================================================================
# # method 1: use dynamic_rnn
# decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
#     decoder_cell, decoder_inputs_embedded,
# 
#     initial_state=encoder_final_state,
# 
#     dtype=tf.float32, time_major=True, scope="plain_decoder",
# )
# =============================================================================

# method 2: use seq2seq
decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')

helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=decoder_inputs_embedded,
        sequence_length=decoder_inputs_length,
        time_major=True)

decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=helper,
        initial_state=encoder_final_state)

decoder_outputs, decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=True)#,
        #impute_finished=True)
        
decoder_outputs = decoder_outputs.rnn_output
```
