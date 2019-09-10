# Import dependencies
import numpy as np
import tensorflow as tf

# Defining the model class
class LongShortTermMemoryModel:
    def __init__(self, encoding_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [])  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, 26])  # Shape: [batch_size, max_time, encoding_size]
        self.y = tf.placeholder(tf.float32, [None, None, 10])  # Shape: [batch_size, max_time, encoding_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, encoding_size]))
        b = tf.Variable(tf.random_normal([encoding_size]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.einsum('bts,se->bte', lstm, W), b)  # b: batch, t: time, s: state, e: encoding

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

# Creating the training set
results = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 'cat'
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 'alien'
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 'ghost'
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 'bat'
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 'panda'
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 'kiwi'
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 'wine'
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 'knife'
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 'star'
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 'rat'
]

# Encodes all letters a-z as input vector for model
characters = dict()

for i in range(97, 97+26):
    characters[chr(i)] = [0 if not j == i-97 else 1 for j in range(26)]


encoding_size = 10

index_to_emoji = ['😺', '👽', '👻', '🦇', '🐼', '🥝', '🍷', '🔪', '⭐', '🐀']

x_train = [
    [characters[x] for x in 'cat'],
    [characters[x] for x in 'alien'],
    [characters[x] for x in 'ghost'],
    [characters[x] for x in 'bat'],
    [characters[x] for x in 'panda'],
    [characters[x] for x in 'kiwi'],
    [characters[x] for x in 'wine'],
    [characters[x] for x in 'knife'],
    [characters[x] for x in 'star'],
    [characters[x] for x in 'rat'],
]
y_train = [
    [results[0]],
    [results[1]],
    [results[2]],
    [results[3]],
    [results[4]],
    [results[5]],
    [results[6]],
    [results[7]],
    [results[8]],
    [results[9]],
]

# Creating model and initializing session
model = LongShortTermMemoryModel(encoding_size)

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

# Initialize model.in_state
zero_state = session.run(model.in_state, {model.batch_size: 1})


# Training the model
# Trained an equal ammount of times on all emojis
for epoch in range(500):
    session.run(minimize_operation, {model.batch_size: 1, model.x: [x_train[epoch%10]], model.y: [y_train[epoch%10]], model.in_state: zero_state})


# Function for predicting emoji from text
def predict(text):
    y = ''
    state = session.run(model.in_state, {model.batch_size: 1})

    for c in text:
        y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[characters[c]]], model.in_state: state})

    return index_to_emoji[y[0].argmax()]


# Test program. Let's user input a string and prints the resulting emoji.
text = input('Please input a word:\t').split()[0].lower()
while not text == 'exit':
    print("Program returns:", predict(text))
    text = input('Please input a word:\t').split()[0].lower()
print('\nExiting program.')


session.close()