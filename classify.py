############################################
# Loading required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats

plt.style.use('ggplot')

#### Defining some helper functions
# Reading function
def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

# Normalising function
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

# Plotting function
def plot_axis(ax, x, y, title):
    ax.plot(x,y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
				
# Plotting function
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

# Read the data set and normalise each of the accelerometer component
dataset = read_data('actitracker_modified2.txt')
dataset.dropna(axis=0, how='any', inplace= True)
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])

# Visualize each component of accelerometer for different activities
for activity in np.unique(dataset["activity"]):
	subset = dataset[dataset["activity"] == activity][:180]
	plot_activity(activity, subset)

def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size // 2)

def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    it_count = 0
    for (start, end) in windows(data['timestamp'], window_size):
        it_count+=1
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(dataset['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x,y,z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
            #print("segment_sig_count: ", it_count)
    return segments, labels

segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
reshaped_segments = segments.reshape(len(segments), 1, 90, 3)

# Divide data into training and testing set (70/30) randomly
train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

##########################################
#CNN model
# Define some helper functions and configuration variable for our CNN model

input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 20

total_batches = reshaped_segments.shape[0] // batch_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x,W, [1,1,1,1], padding='VALID')

def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth*num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))

def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')

# Tensorflow placeholders for input and output data				
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

c = apply_depthwise_conv(X, kernel_size, num_channels, depth)
p = apply_max_pool(c, 20, 2)
c = apply_depthwise_conv(p, 6, depth*num_channels, depth//10)

shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1]*shape[2]*shape[3]])

# The tanh function is used as non-linearity
f_weights_l1 = weight_variable([shape[1]*shape[2]*depth*num_channels*(depth//10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

# the Softmax layer is defined to output probabilities of the class labels
out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

# The negative log-likelihood cost function will be minimised using stochastic gradient descent optimizer
loss = -tf.reduce_sum(Y*tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)

# Training the model
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        for b in range(total_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)
        print("Epoch: ",epoch," Training Loss: ",c," Training Accuracy: ",
                sess.run(accuracy, feed_dict={X: train_x, Y: train_y}))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
