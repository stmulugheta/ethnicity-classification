from preprocess import (
    input_setup,
    read_data
)

import numpy as np
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from random import shuffle

class ZGLFace(object):
    def __init__(self, sess, image_size, label_num, batch_size, channel_num, checkpoint_dir):
        self.name = 'ZFLFace'
        self.sess = sess
        self.image_size = image_size
        self.label_num = label_num
        self.batch_size = batch_size
        self.channel_num = channel_num
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.channel_num], name='images')
        self.labels = tf.placeholder(tf.int64, [None, self.label_num], name='labels')

        self.weights = {
        'w1': tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.01), name='w1'),
        'w2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01), name='w2'),
        'w3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01), name='w3'),
        'wfc1': tf.Variable(tf.truncated_normal([7 * 7 * 128, 256], stddev=0.01), name='wfc1'),
        'wfc2': tf.Variable(tf.truncated_normal([256, 256], stddev=0.01), name='wfc2'),
        'wfc3': tf.Variable(tf.truncated_normal([256, self.label_num], stddev=0.01), name='wfc3')
        }

        self.biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[32]), name="b1"),
        'b2': tf.Variable(tf.constant(0.1, shape=[64]), name="b2"),
        'b3': tf.Variable(tf.constant(0.1, shape=[128]), name="b3"),
        'bfc1': tf.Variable(tf.constant(0.1, shape=[256]), name="bfc1"),
        'bfc2': tf.Variable(tf.constant(0.1, shape=[256]), name="bfc2"),
        'bfc3': tf.Variable(tf.constant(0.1, shape=[self.label_num]), name="bfc3")
        }

        self.pred = self.model()

        # Cross Entropy
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,labels=self.labels))
        
        self.ce_cost = tf.reduce_mean(self.cross_entropy)
        
        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.weights['wfc3']) + tf.nn.l2_loss(self.biases['bfc3']) +
              tf.nn.l2_loss(self.weights['wfc2']) + tf.nn.l2_loss(self.biases['bfc2']) +
              tf.nn.l2_loss(self.weights['wfc1']) + tf.nn.l2_loss(self.biases['bfc1']) +
              tf.nn.l2_loss(self.weights['w3']) + tf.nn.l2_loss(self.biases['b3']) +
              tf.nn.l2_loss(self.weights['w2']) + tf.nn.l2_loss(self.biases['b2']) +
              tf.nn.l2_loss(self.weights['w1']) + tf.nn.l2_loss(self.biases['b1'])
             )

        # Add the regularization term to the loss.
        self.cross_entropy += 5e-4 * regularizers
        
        self.prediction=tf.nn.softmax(self.pred)
        
        # Summary for loss
        self.loss_summary = tf.summary.scalar('Train Loss', self.cross_entropy)

        # Merge summaries
        self.summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver()
        
    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        mxp1 = tf.nn.max_pool(conv1, [1,3,3,1], strides=[1,2,2,1], padding='SAME')
        lrn1 = tf.nn.local_response_normalization(mxp1, alpha=0.0001, beta=0.75)
        conv2 = tf.nn.relu(tf.nn.conv2d(lrn1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        mxp2 = tf.nn.max_pool(conv2, [1,3,3,1], strides=[1,2,2,1], padding='VALID')
        lrn2 = tf.nn.local_response_normalization(mxp2, alpha=0.0001, beta=0.75)
        conv3 = tf.nn.conv2d(lrn2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']
        mxp3 = tf.nn.max_pool(conv3, [1,3,3,1], strides=[1,2,2,1], padding='VALID')
        mxp1_flat = tf.reshape(mxp3, [-1, 7 * 7 * 128])
        fc1 = tf.nn.relu(tf.matmul(mxp1_flat, self.weights['wfc1']) + self.biases['bfc1'])
        dfc1 = tf.nn.dropout(fc1, 0.5)
        fc2 = tf.nn.relu(tf.matmul(dfc1, self.weights['wfc2']) + self.biases['bfc2'])
        dfc2 = tf.nn.dropout(fc2, 0.7)
        fc3 = (tf.matmul(dfc2, self.weights['wfc3']) + self.biases['bfc3'])
        return fc3
    
    def save(self, checkpoint_dir, step):
        model_name = "instaface.model"
        model_dir = "instaface"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        
    def plot_example_errors(cls_pred, correct):
        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = data.valid.images[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = data.valid.cls[incorrect]

        # Plot the first 9 images.
        plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])
        
    def plot_confusion_matrix(cls_pred):
        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the true classifications for the test-set.
        cls_true = data.valid.cls

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=cls_pred)

        # Print the confusion matrix as text.
        print(cm)

        # Plot the confusion matrix as an image.
        plt.matshow(cm)

        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()
        
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    
    def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
        # Calculate the accuracy on the training-set.
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))
    
    def train(self, config):
        print config
        input_setup(self.sess, config)
        
        if config.is_train:
            train_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            train_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        
        # Load data
        train_data, train_label = read_data(train_data_dir)
        
        # Shuffle data
        ind_list = [i for i in range(len(train_data))]
        shuffle(ind_list)
        train_data  = train_data[ind_list, :,:,:]
        train_label = train_label[ind_list,]
        
        # Adam
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.cross_entropy)
        
        y_true_cls = tf.argmax(self.labels, 1)
        
        # Operation comparing prediction with true label
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        #tf.global_variables_initializer()
        tf.initialize_all_variables().run()
        
        counter = 0
        start_time = time.time()
        if config.is_train:
            graph = tf.get_default_graph()
            summary_writer = tf.summary.FileWriter('logs/', graph)
            print("Training...")


            for ep in xrange(config.epoch):
            # Run by batch images
                batch_idxs = len(train_data) // config.batch_size
                for idx in xrange(0, batch_idxs):
                    train_batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                    train_batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

                    counter += 1
                    _, err, summary = self.sess.run([self.train_op, self.cross_entropy, self.summaries], feed_dict={self.images: train_batch_images, self.labels: train_batch_labels})

                    # Save summaries to log file
                    summary_writer.add_summary(summary, counter)
                    summary_writer.flush()

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                          % ((ep+1), counter, time.time()-start_time, err))
                    if counter % 100 == 0:
                        train_accuracy = self.sess.run(self.accuracy, feed_dict={self.images: train_batch_images, self.labels: train_batch_labels})
                        print('Step {:5d}: training accuracy {:g}'.format(counter, train_accuracy))
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
            
            # Validation accuracy at the end
            valid_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "valid.h5")
            valid_data, valid_label = read_data(valid_data_dir)
            ind_list = [i for i in range(len(valid_data))]
            shuffle(ind_list)
            valid_data  = valid_data[ind_list, :,:,:]
            valid_label = valid_label[ind_list,]
            test_accuracy = self.sess.run(self.accuracy, feed_dict={self.images: valid_data, self.labels: valid_label})
            print('Test accuracy {:g}'.format(test_accuracy))
            
            save_path = self.saver.save(self.sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)
        else:
            # Restore model weights from previously saved model
            self.saver.restore(self.sess, "model.ckpt")
            print("Model restored from file: %s" % save_path)
            
            graph = tf.get_default_graph()
            summary_writer = tf.summary.FileWriter('logs/', graph)
            
            print("Testing...")
            
            for ep in xrange(config.epoch):
            # Run by batch images
                batch_idxs = len(train_data) // config.batch_size
                for idx in xrange(0, batch_idxs):
                    train_batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                    train_batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

                    counter += 1
                    _, err, summary = self.sess.run([self.train_op, self.cross_entropy, self.summaries], feed_dict={self.images: train_batch_images, self.labels: train_batch_labels})

                    # Save summaries to log file
                    summary_writer.add_summary(summary, counter)
                    summary_writer.flush()

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                          % ((ep+1), counter, time.time()-start_time, err))
                        train_accuracy = self.sess.run(self.accuracy, feed_dict={self.images: train_batch_images, self.labels: train_batch_labels})
            