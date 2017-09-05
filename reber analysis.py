"""Training an saving of a Tensorflow RNN model using LSTM cells
 to differentiate between true and false embedded reber grammars"""

import rebergood as rbg
import reberbad as rbb
import numpy as np
import tensorflow as tf

dataset_size = 10000
true_percent = 0.5
true_count = int(dataset_size*true_percent)
false_count = dataset_size - true_count

# Generate the dataset of strings and corresponding labels
dataset = np.zeros(shape=[0, 2])
for i in range(true_count):
    unit = np.array((rbg.embedded_reber(), 0))
    dataset = np.vstack((dataset, unit))

for j in range(false_count):
    unit = np.array((rbb.embedded_reber_bad(), 1))
    dataset = np.vstack((dataset, unit))

rand_indices = np.random.permutation(dataset_size)
dataset = dataset[rand_indices]

# Split into strings and labels
X_dat = dataset[:, 0]
y_dat = dataset[:, 1]
max_len = max([len(i) for i in X_dat])

# converts string into a sparse matrix
def sparserizer(word, length= -20):
    if length == -20:
        length = len(word)
    sparse = np.zeros(shape=[7, length])

    for i in range(len(word)):
        letter = word[i]
        sparsecol = sparse[:, i]
        if letter == "B":
            sparsecol[0] = 1
        if letter == "E":
            sparsecol[1] = 1
        if letter == "P":
            sparsecol[2] = 1
        if letter == "S":
            sparsecol[3] = 1
        if letter == "T":
            sparsecol[4] = 1
        if letter == "V":
            sparsecol[5] = 1
        if letter == "X":
            sparsecol[6] = 1

    return sparse


# converts a batch of strings into a sparse matrix
def batch_sparse(dataset, batch_size, batch_num):
    words = X_dat[batch_size * batch_num:batch_size * batch_num + batch_size]
    y = y_dat[batch_size * batch_num:batch_size * batch_num + batch_size]
    lens = [len(s) for s in words]
    length = max_len
    sparse_table = np.zeros(shape=[0, 7])
    for word in words:
        sparse_table = np.vstack((sparse_table, sparserizer(word, length).transpose()))

    sparse_table = np.reshape(sparse_table, [-1, length, 7])
    return sparse_table, y, lens


# Building the Tensorflow Graph

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, max_len, 7])
y = tf.placeholder(tf.int32, [None])
lens = tf.placeholder(tf.int32, [None])
basic_cell = tf.contrib.rnn.GRUCell(num_units=170)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=lens)

logits = tf.layers.dense(states, 2)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Preparing the execution phase
n_epochs = 100
batch_size = 40
num_batches = dataset_size // batch_size

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(num_batches - 2):
            X_batch, y_batch, lens_batch = batch_sparse(X_dat, batch_size, iteration)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, lens: lens_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, lens: lens_batch})
        X_test, y_test, lens_test = batch_sparse(X_dat, batch_size, num_batches - 1)
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test, lens: lens_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    saver.save(sess, "my_reber_model")
