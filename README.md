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


将前一个的输出作为后一个decoder的输入

``` python
# 传给CustomHelper的三个函数
def initial_fn():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
    return initial_elements_finished, initial_input

def sample_fn(time, outputs, state):
    # 选择logit最大的下标作为sample
    outputs = tf.matmul(outputs, W) + b
    prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
    return prediction_id

def next_inputs_fn(time, outputs, state, sample_ids):
    # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
    next_input = tf.nn.embedding_lookup(embeddings, sample_ids)
    # pay attention to here
    # check BasicDecoder and dynamic_decode, since initial_time is set to 0 at the beginning, so we do time + 1
    elements_finished = (time + 1 >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
    next_state = state
    return elements_finished, next_inputs, next_state

my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)


decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=my_helper,
        initial_state=encoder_final_state)

decoder_outputs, decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=True,
        # pay attention to here
        impute_finished=True)

decoder_outputs = decoder_outputs.rnn_output
```

具体调用三个传给CustomHelper的三个函数可参考https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py

``` python
class BasicDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize BasicDecoder.
    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.
    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer
```
``` python
def initialize(self, name=None):
    """Initialize the decoder.
    Args:
      name: Name scope for any created operations.
    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.
    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.
    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)
```
