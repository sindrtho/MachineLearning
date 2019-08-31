import numpy as np
import tensorflow as tf

(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train_, (-1, 28, 28, 1))  # tf.nn.conv2d takes 4D arguments
y_train = np.zeros((y_train_.size, 10))
y_train[np.arange(y_train_.size), y_train_] = 1

batches = 600  # Divide training data into batches to speed up optimization
x_train_batches = np.split(x_train, batches)
y_train_batches = np.split(y_train, batches)

x_test = np.reshape(x_test_, (-1, 28, 28, 1))
y_test = np.zeros((y_test_.size, 10))
y_test[np.arange(y_test_.size), y_test_] = 1


class ConvolutionalNeuralNetworkModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        W1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))  # 5x5 filters, 1 in-channel, 32 out-channels
        b1 = tf.Variable(tf.random_normal([32]))
        W2 = tf.Variable(tf.random_normal([14 * 14 * 32, 10]))  # (width / 2) * (height / 2) * 32. Divided by 2 due to pooling
        b2 = tf.Variable(tf.random_normal([10]))

        # Model operations
        conv1 = tf.nn.bias_add(tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME'), b1)  # Using builtin function for adding bias
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Logits
        logits = tf.nn.bias_add(tf.matmul(tf.reshape(pool1, [-1, 14 * 14 * 32]), W2), b2)

        # Predictor
        f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))


model = ConvolutionalNeuralNetworkModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.AdamOptimizer(0.001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(20):
    for batch in range(batches):
        session.run(minimize_operation, {model.x: x_train_batches[batch], model.y: y_train_batches[batch]})

    print("epoch", epoch)
    print("accuracy", session.run(model.accuracy, {model.x: x_test, model.y: y_test}))

session.close()
