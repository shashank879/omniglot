from __future__ import print_function
import tensorflow as tf
import numpy as np
from datetime import datetime
import os, time, math
import ops
from logger import logger
from sklearn.utils import shuffle
import glob
from PIL import Image


np.set_printoptions(precision=4)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AE_classifier():

    IM_SIZE = 105

    def __init__(self, classes, dims=8, batch_size=64, epochs=40, summarize=False, name='default'):
        """ Inititialize the variables and start graph construction

            Args:
                classes (list): List of the names of the classes
                dims (int): The number of filters in the first convolution layer, the subsequent layers have filters as a multiple of this
                batch_size (int): The batch size to use while training
                epochs (int): Number of epochs to train for
                summarize (boolean): Should the model output the summary of losses
                name (string): When working with different variants of the model, it can be used to store them in differnt directories. This allows for better testing of the variants
        """
        self.classes = classes
        self.dims = dims
        self.n_classes = len(classes)
        self.batch_size = batch_size
        self.epochs = epochs
        self.summarize = summarize
        self.name = name

        # Build directory structure
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logpath = os.path.join(self.__class__.__name__, name, 'logs', self.run_id)
        self.outpath = os.path.join(self.__class__.__name__, name, 'outputs', self.run_id)
        self.saves_path = os.path.join(self.__class__.__name__, name, 'saves')
        make_sure_path_exists(self.logpath)
        make_sure_path_exists(self.outpath)

        # Make session
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        # Build the tensorflow graph
        self.build()
        print('Run id:', self.run_id)

    def load(self):
        """ Load the model defined by the variable 'name'
        """
        print("loading {} model...".format(self.name))
        chkpt = tf.train.get_checkpoint_state(self.saves_path)

        if chkpt and chkpt.model_checkpoint_path:
            chkpt_name = os.path.basename(chkpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.saves_path, chkpt_name))
            logger.info("[*] Successfully restored... ", chkpt_name)
            return True
        else:
            logger.info("[*] Failed to find the checkpoint...")
            return False

    def save(self, step):
        """ Save the model in the directory 'name'
        """
        make_sure_path_exists(self.saves_path)
        self.saver.save(self.sess,
                        os.path.join(self.saves_path, self.__class__.__name__ + '.ckpt'),
                        global_step=step)

    def _build_graph(self, training=True, reuse=False):
        """ Function that constructs the actual computation graph

            Args:
                training (boolean): Defines the graph mode, allows modifications to the same graph during inference
                reuse (boolean): Defines if the graph variables are to be reused
        """
        with tf.variable_scope('graph', reuse=reuse):
            # Options for regularizers. It was found 
            norm_fn = None
            norm_params = {}
            # norm_fn = tf.contrib.layers.batch_norm
            # norm_params = {'updates_collections': None, 'is_training': training}

            # The Encoder that outputs the feature representations of the image
            with tf.variable_scope('encoder'):
                c0 = ops.multi_kernel_conv2d(self.preprocessed_img, self.dims, stride=2, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='c0')
                print(c0.name, c0.get_shape())
                c1 = ops.multi_kernel_conv2d(c0, self.dims * 2, stride=2, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='c1')
                print(c1.name, c1.get_shape())
                c2 = ops.multi_kernel_conv2d(c1, self.dims * 4, stride=2, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='c2')
                print(c2.name, c2.get_shape())
                c3 = ops.multi_kernel_conv2d(c2, self.dims * 8, stride=2, activation_fn=None, scope='c3')
                print(c3.name, c3.get_shape())

            """ The decoder that reconstructs the original image from the encoded feature representations
                Different variants of the decoder were tested.
            """
            with tf.variable_scope('decoder'):
                ## 1. Simple deconvolutions with linear activations
                # d0 = tf.contrib.layers.conv2d_transpose(c3, self.dims * 3 * 4, 3, stride=2, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='d0')
                # print(d0.name, d0.get_shape())
                # d1 = tf.contrib.layers.conv2d_transpose(d0, self.dims * 3 * 2, 3, stride=2, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='d1')
                # print(d1.name, d1.get_shape())
                # d2 = tf.contrib.layers.conv2d_transpose(d1, self.dims * 3 * 1, 3, stride=2, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='d2')
                # print(d2.name, d2.get_shape())
                # d3 = tf.contrib.layers.conv2d_transpose(d2, 1, 3, stride=2, activation_fn=None, scope='d3')
                # print(d3.name, d3.get_shape())

                ## 2. Multi-kernel deconvolution
                # d0 = ops.multi_kernel_conv2d_transpose(c3, self.dims * 4, stride=2, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='d0')
                # print(d0.name, d0.get_shape())
                # d1 = ops.multi_kernel_conv2d_transpose(d0, self.dims * 2, stride=2, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='d1')
                # print(d1.name, d1.get_shape())
                # d2 = ops.multi_kernel_conv2d_transpose(d1, self.dims * 1, stride=2, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='d2')
                # print(d2.name, d2.get_shape())
                # # d3 = ops.multi_kernel_conv2d_transpose(d2, 1, stride=2, activation_fn=None, scope='d3')
                # # print(d3.name, d3.get_shape())
                # d3 = tf.contrib.layers.conv2d_transpose(d2, 1, 5, stride=2, activation_fn=None, scope='d3')
                # print(d3.name, d3.get_shape())

                ## 3. Residual based upscaling (for implementation details please have a look at the ops.py file)
                d0 = ops.resnet_block_transpose(c3, self.dims * 3 * 4, kernel_size=3, stride=2, scope='d0')
                print(d0.name, d0.get_shape())
                d1 = ops.resnet_block_transpose(d0, self.dims * 3 * 2, kernel_size=3, stride=2, scope='d1')
                print(d1.name, d1.get_shape())
                d2 = ops.resnet_block_transpose(d1, self.dims * 3 * 1, kernel_size=3, stride=2, scope='d2')
                print(d2.name, d2.get_shape())
                d3 = ops.resnet_block_transpose(d2, 1, kernel_size=3, stride=2, activation_fn=None, scope='d3')
                print(d3.name, d3.get_shape())

            # Classifier that outputs logits over the number of classes.
            with tf.variable_scope('classifier'):
                n = 8 * self.dims * 3 * 16
                h0 = tf.reshape(c3, shape=[-1, n], name='h0')
                print(h0.name, h0.get_shape())
                h1 = tf.layers.dense(h0, n, activation=tf.nn.leaky_relu, name='h1')
                print(h1.name, h1.get_shape())
                h2 = tf.layers.dense(h0, n, activation=tf.nn.leaky_relu, name='h2')
                print(h2.name, h2.get_shape())
                h3 = tf.layers.dense(h2, self.n_classes, name='h3')
                print(h3.name, h3.get_shape())
        return c3, d3, h3

    def build(self):
        """ The function performs the other required steps for running the graph:
            1. Constructs the placeholders
            2. Preprocesses the images
            3. Instructs to construct the graph
            4. Makes loss functions and summaries
            5. Builds optimizers and other ops
        """
        with self.graph.as_default(), tf.variable_scope(self.__class__.__name__):
            ## 1. Construct the placeholders (since the image is b/w, the channel dimension is not present)
            self.X = tf.placeholder(tf.float32, shape=[None, self.IM_SIZE, self.IM_SIZE], name='input_images')
            self.Y = tf.placeholder(tf.int32, shape=[None], name='input_labels')

            ## 2. Preprocesses the images
            reshaped_img = tf.reshape(self.X, shape=[-1, self.IM_SIZE, self.IM_SIZE, 1])
            resized_img = tf.image.resize_images(reshaped_img, [64, 64])
            self.preprocessed_img = resized_img

            ## 3. Build train and inference graphs
            with tf.name_scope('train'):
                _, recons, cl_logits = self._build_graph()
                preds = tf.squeeze(tf.nn.top_k(cl_logits).indices, axis=1)
            with tf.name_scope('inference'):
                self.features, test_recons, test_cl_logits = self._build_graph(training=False, reuse=True)
                self.reconstruction = tf.squeeze(tf.sigmoid(test_recons), axis=3)
                self.pred_class = tf.nn.softmax(test_cl_logits)

            ## 4. Build loss functions
            one_hot_labels = tf.one_hot(self.Y, self.n_classes)
            # self.recons_loss = tf.reduce_mean(tf.squared_difference(recons, self.preprocessed_img), name='recons_loss')
            self.recons_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.preprocessed_img, logits=recons), name='recons_loss')
            self.cl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=cl_logits), name='cl_loss')

            ## 5. Build variables for calculating metrics
            n_correct = tf.reduce_sum(tf.cast(tf.equal(self.Y, preds), dtype=tf.float32), axis=0)
            n_total = tf.cast(tf.shape(self.Y)[0], dtype=tf.float32)
            self.accuracy = n_correct / n_total

            ## 6. Build summary ops
            tf.summary.scalar('reconstruction_loss', self.recons_loss)
            tf.summary.scalar('classification_loss', self.cl_loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.image('original', self.preprocessed_img)
            tf.summary.image('reconstruction', recons)

            ## 7. Collect train variables and check if some variables are left untrained
            vars = tf.trainable_variables()
            recons_vars = [v for v in vars if 'encoder' in v.name or 'decoder' in v.name]
            cl_vars = [v for v in vars if 'classifier' in v.name]
            untraining_vars = [v for v in vars if ((v not in recons_vars) and (v not in cl_vars))]
            if len(untraining_vars) > 0:
                print(untraining_vars)
                raise Exception('Some trainable variables are not being trained')

            ## 8. Create train ops for built losses and varaibles, and other required ops
            self.opt = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.recons_loss + self.cl_loss, var_list=recons_vars + cl_vars)
            self.saver = tf.train.Saver(max_to_keep=1)
            self.init_op = tf.global_variables_initializer()
            # self.init_op = tf.group(tf.global_variables_initializer(), validation_metrics_init_op)
            self.summ_op = tf.summary.merge_all()

    def train(self, images, labels):
        """ Starts the training using images and corresponding labels
            Args:
                images (list): The list of images of characters
                labels (list): The list of corresponding labels of images
        """
        assert len(images) == len(labels), 'Lengths of input and output sequences must be same'
        logger.info('Training for {}...'.format(self.name))

        ## Initialize the graph and build a summary writer if required
        with self.graph.as_default():
            self.sess.run(self.init_op)
            if self.summarize:
                writer = tf.summary.FileWriter(self.logpath, graph=self.sess.graph)

        train_st_time = time.time()
        counter = 0
        ## Calculate number of batches allowing for incomplete last batch
        n_batches = int(math.ceil(len(images) / float(self.batch_size)))
        print('Starting train loop...\n')
        for epoch in range(self.epochs):
            ## Shuffle data before each epochs
            images, labels = shuffle(images, labels)
            epoch_st_time = time.time()
            for i in range(n_batches):
                step = i + 1
                st_time = time.time()
                xbatch = images[self.batch_size * i:self.batch_size * (i + 1)]
                ybatch = labels[self.batch_size * i:self.batch_size * (i + 1)]

                feed_dict = {self.X: xbatch,
                             self.Y: ybatch}
                if self.summarize and (counter % 2 == 0):
                    _, recons_loss, cl_loss, acc, summ_str = self.sess.run([self.opt, self.recons_loss, self.cl_loss, self.accuracy, self.summ_op], feed_dict=feed_dict)
                    writer.add_summary(summ_str, counter)
                else:
                    _, recons_loss, cl_loss, acc = self.sess.run([self.opt, self.recons_loss, self.cl_loss, self.accuracy], feed_dict=feed_dict)
                step_time = time.time() - st_time
                print('Epoch: {} Step: {}/{}, recons_loss: {:0.4f}, cl_loss: {:0.4f}, accuracy: {:0.4f}; step_time: {:0.2f}s, elapsed time: {:.02f}m, remaining_time: {:0.2f}m'.format(epoch + 1, step, n_batches, recons_loss, cl_loss, acc, step_time, (time.time() - epoch_st_time) / 60, (step_time * (n_batches - step) / 60)), end='\r')
                counter += 1
            print()
        ## Save after training is completes
        self.save(counter)
        print('{} training complete in... {:0.2f}m'.format(self.name, (time.time() - train_st_time) / 60))

    def feature_distance(self, img1, img2):
        """ Function to calculate the similarity of the images based on feature distance
            Args:
                img1 (numpy.ndarray): 2D array of shape HxW
                img2 (numpy.ndarray): 2D array of shape HxW
            Returns:
                The feature distance computed as mean squared distance
        """
        features = self.sess.run(self.features, feed_dict={self.X: [img1, img2]})
        return np.square(features[0] - features[1]).mean()

    def img_reconstruction(self, imgs):
        """ Outputs the reconstructed image from the decoder. Can be used as a metric to test how much the network understands the image.
            Args:
                imgs (list): List of images (HxW) to reconstruct
            Returns:
                List of reconstructions of size (64x64)
        """
        recons = self.sess.run(self.reconstruction, feed_dict={self.X: imgs})
        return recons


def read_data(folder_path):
    """ Function that reads the images recursively in the directory (omniglot data directory structure only). And uses folder path as the image label
        Args:
            folder_path (string): Path to the data folder
        Returns:
            Images: List of images found
            Labels: List of labels corresponding to the images
            classes: List of strings, as total different classes found
    """
    imgs = []
    labels = []
    classes = []
    for im_path in glob.glob(folder_path + '/*/*/*.png'):
        img = np.asarray(Image.open(im_path), dtype=np.float32)
        label = im_path[:-12]
        imgs.append(img)
        if label not in classes:
            classes.append(label)
        labels.append(classes.index(label))
    return imgs, labels, classes


def make_sure_path_exists(path):
    """ Util function that creates the specified directory if required
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    imgs, labels, classes = read_data('../images_background')
    ae_cl = AE_classifier(classes, summarize=True, name='default')

    # Train a new model or reload previously stored model
    ae_cl.train(imgs, labels)
    # ae_cl.load()

    print('Done!\n\n')
