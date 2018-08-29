#we give encoder input sequence like 'hello how are you', we take the last hidden state and feed to decoder and it
#will generate a decoded value. we compare that to target value, if translation would be 'bonjour ca va' and minimize
#the difference by optimizing a loss function

#in this case we just want to encode and decode the input successfully

#bidirectional encoder
#We will teach our model to memorize and reproduce input sequence.
#Sequences will be random, with varying length.
#Since random sequences do not contain any structure,
#model will not be able to exploit any patterns in data.
#It will simply encode sequence in a thought vector, then decode from it.
#this is not about prediction (end goal), it's about understanding this architecture

#this is an encoder-decoder architecture. The encoder is bidrectional so
#it It feeds previously generated tokens during training as inputs, instead of target sequence.

import math
import numpy as np
import tensorflow as tf #machine learningt
import helpers #for formatting data into batches and generating random sequence data
import process

import time
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Sit back and relax, it will take some time to train the model...")

with open('config.json') as f:
    config = json.load(f)

dataset_file = config["dataset_path"]
model_path = config["model_save"]

tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session

#First critical thing to decide: vocabulary size.
#Dynamic RNN models can be adapted to different batch sizes
#and sequence lengths without retraining
#(e.g. by serializing model parameters and Graph definitions via tf.train.Saver),
#but changing vocabulary size requires retraining the model.

PAD = 0
EOS = 1

vocab_size = config["vocabulary_size"]
print("Vocabulary size", vocab_size)

input_embedding_size = config["input_embedding_size"] #character length

encoder_hidden_units = config["encoder_hidden_units"] #num neurons
decoder_hidden_units = encoder_hidden_units * 2 #in original paper, they used same number of neurons for both encoder
#and decoder, but we use twice as many so decoded output is different, the target value is the original input
#in this example

#input placehodlers
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
#contains the lengths for each of the sequence in the batch, we will pad so all the same
#if you don't want to pad, check out dynamic memory networks to input variable length sequences
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

#randomly initialized embedding matrrix that can fit input sequence
#used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
#reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

#this thing could get huge in a real world application
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple

encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )

#Concatenates tensors along one dimension.
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

#letters h and c are commonly used to denote "output value" and "cell state".
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#Those tensors represent combined internal state of the cell, and should be passed together.

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

decoder_cell = LSTMCell(decoder_hidden_units)

#we could print this, won't need
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

decoder_lengths = encoder_inputs_length + 3
# +2 additional steps, +1 leading <EOS> token for decoder inputs

#manually specifying since we are going to implement attention details for the decoder in a sec
#weights
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
#bias
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

#create padded inputs for the decoder from the word embeddings

#were telling the program to test a condition, and trigger an error if the condition is false.
assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

#manually specifying loop function through time - to get initial cell state and input to RNN
#normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

#we define and return these values, no operations occur here
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded
    #last time steps cell state
    initial_cell_state = encoder_final_state
    #none
    initial_cell_output = None
    #none
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)


# attention mechanism --choose which previously generated token to pass as input in the next timestep
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        # dot product between previous ouput and weights, then + biases
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        # Logits simply means that the function operates on the unscaled output of
        # earlier layers and that the relative scale to understand the units is linear.
        # It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities
        # (you might have an input of 5).
        # prediction value at current time step

        # Returns the index with the largest value across axes of a tensor.
        prediction = tf.argmax(output_logits, axis=1)
        # embed prediction for the next input
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended

    # Computes the "logical and" of elements across dimensions of a tensor.
    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    # Return either fn1() or fn2() based on the boolean predicate pred.
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)

    # set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

#Creates an RNN specified by RNNCell cell and loop function loop_fn.
#This function is a more primitive version of dynamic_rnn that provides more direct access to the
#inputs each iteration. It also provides more control over when to start and finish reading the sequence,
#and what to emit for the output.
#ta = tensor array
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()


#to convert output to human readable prediction
#we will reshape output tensor

#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#reduces dimensionality
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
#flettened output tensor
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
#pass flattened tensor through decoder
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
#prediction vals
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
#final prediction
decoder_prediction = tf.argmax(decoder_logits, 2)

#cross entropy loss
#one hot encode the target values so we don't rank just differentiate
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

#loss function
loss = tf.reduce_mean(stepwise_cross_entropy)
#train it
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

batch_size = config["batch_size"]


def get_dataset_iterator(partition):
    file_names = [dataset_file+partition+".incorrect.txt", dataset_file+partition+".correct.txt"]
    trainX = tf.data.TextLineDataset(file_names[0])
    trainY = tf.data.TextLineDataset(file_names[1])
    train_dataset = tf.data.Dataset.zip((trainX, trainY)).batch(batch_size).repeat()

    iterator = train_dataset.make_initializable_iterator()

    next_element = iterator.get_next()
    return next_element, iterator


def process_data(batches):
    data_list = []
    for e in batches:
        data_list.append(np.ndarray.tolist(np.fromstring(e, dtype=int, sep=' ')))

    return data_list


train_next_batch, train_iterator = get_dataset_iterator("train")
valid_next_batch, valid_iterator = get_dataset_iterator("valid")
test_next_batch, test_iterator = get_dataset_iterator("test")

sess.run(train_iterator.initializer)
sess.run(valid_iterator.initializer)
sess.run(test_iterator.initializer)


def next_feed(partition):
    if partition == "train":
        sample, target = sess.run(train_next_batch)
    elif partition == "valid":
        sample, target = sess.run(valid_next_batch)
    else:
        sample, target = sess.run(test_next_batch)

    sample = process_data(sample)
    target = process_data(target)

    encoder_inputs_, encoder_input_lengths_ = helpers.batch(sample)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in target]
    )

    if partition == "test":
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_,
        }, target
    else:
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_,
        }


start_time = time.time()
batch_time = start_time
saver = tf.train.Saver()

accuracies = [0]
a_epoch = [0]
train_losses = []
tl_epoch = []
validation_losses = []
vl_epoch = []
times = [0]
t_epoch = [0]

try:
    for epoch in range(1, config["n_epoch"]):
        epoch_loss = []
        print("\nepoch {}".format(epoch))

        print("  training ", end="", flush=True)
        for ba in range(config["batch_in_epoch"]):
            fd = next_feed("train")
            _, l = sess.run([train_op, loss], fd)
            epoch_loss.append(l)

            if ba % math.ceil((config["batch_in_epoch"] * 10)/100) == 0:
                print(".", end="", flush=True)
        train_losses.append(np.average(epoch_loss))
        tl_epoch.append(epoch)
        print("\n    epoch loss {}".format(np.average(epoch_loss)))

        if epoch % config["validation_point"] == 0:
            validation_loss = []
            print("  validating ", end="", flush=True)
            for ba in range(config["validation_batch"]):
                fd = next_feed("train")
                _, l = sess.run([train_op, loss], fd)
                validation_loss.append(l)
                if ba % math.ceil((config["validation_batch"] * 10) / 100) == 0:
                    print(".", end="", flush=True)
            validation_losses.append(np.average(validation_loss))
            vl_epoch.append(epoch)
            print("\n    epoch loss {}".format(np.average(validation_loss)))

        print("  testing ", end="", flush=True)
        epoch_accuracy = []
        for ba in range(config["test_batch"]):
            fd, correct = next_feed("test")

            predict_ = sess.run(decoder_prediction, fd)

            for i, (inp, predicted, c) in enumerate(zip(fd[encoder_inputs].T, predict_.T, correct)):
                size = len(c)
                accuracy = np.sum(predicted[:len(c)] == c)/size
                epoch_accuracy.append(accuracy)

            if ba % math.ceil((config["test_batch"] * 10)/100) == 0:
                print(".", end="", flush=True)

        accuracies.append(np.average(epoch_accuracy))
        a_epoch.append(epoch)
        print("\n    accuracy {}".format(np.average(epoch_accuracy)))

        del epoch_accuracy

        save_path = saver.save(sess, model_path, global_step=epoch)
        print("  Model saved in path: %s" % save_path)
        batch_elapsed_time = time.time() - start_time

        times.append(batch_elapsed_time)
        t_epoch.append(epoch)

        print("  Epoch time: ", time.strftime("%H:%M:%S", time.gmtime(batch_elapsed_time)))
        batch_time = time.time()

        train_stat = {
            "train": {"epoch": tl_epoch, "losses": train_losses},
            "valid": {"epoch": vl_epoch, "losses": validation_losses},
            "accuracy": {"epoch": a_epoch, "accuracies": accuracies},
            "time": {"epoch": t_epoch, "times": times}
        }

        process.save_obj(train_stat, "train_stat")

except KeyboardInterrupt:
    print('training interrupted')

elapsed_time = time.time() - start_time
print("Total time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
