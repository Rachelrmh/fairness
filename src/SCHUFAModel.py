import tensorflow as tf
import tf_helper_functions as tf_helper
import os

class SCHUFAModel:

    def __init__(self, config):
        self.config = config
        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.config["inputs"]], name="x")

        with tf.name_scope("dnn_model"):
            #input layer(hidden 1)
            self.w1 = tf_helper.weights([self.config["inputs"], self.config["hidden1"]], name="w1")
            b1 = tf_helper.bias([self.config["hidden1"]])
            z1 = tf_helper.dense_layer(self.x, self.w1, b1, activation=tf.nn.relu)
            z1_drop = tf.nn.dropout(z1, self.config["dropout_prob"])

            #hidden layer 2
            self.w2 = tf_helper.weights([self.config["hidden1"], self.config["hidden2"]], name="w2")
            b2 = tf_helper.bias([self.config["hidden2"]])
            z2 = tf_helper.dense_layer(z1_drop, self.w2, b2, activation=tf.nn.relu)
            z2_drop = tf.nn.dropout(z2, self.config["dropout_prob"])

            #hidden layer 3
            self.w3 = tf_helper.weights([self.config["hidden2"], self.config["hidden3"]], name="w3")
            b3 = tf_helper.bias([self.config["hidden3"]])
            z3 = tf_helper.dense_layer(z2_drop, self.w3, b3, activation=tf.nn.relu)
            z3_drop = tf.nn.dropout(z3, self.config["dropout_prob"])

            #output layer
            self.w4 = tf_helper.weights([self.config["hidden3"], self.config["outputs"]], name="w4")
            b4 = tf_helper.bias([self.config["outputs"]])
            z4 = tf_helper.dense_layer(z3_drop, self.w4, b4, activation=None)


            '''
            self.logits = z3
            values, indices = tf.nn.top_k(tf.nn.softmax(self.logits))
            self.predictions = indices
            '''
        with tf.name_scope("loss"):
            if(self.config["loss"] == "log_loss"):
                self.y = tf.placeholder(dtype=tf.int32, shape=(None, self.config["outputs"]), name="y")
                self.logits = tf.nn.sigmoid(z4)

                #binarize logits for final predictions
                prediction_thresh = tf.constant(0.5, dtype=tf.float32)
                self.predictions=tf.cast(tf.greater_equal(self.logits, prediction_thresh), tf.float32)
                self.loss = tf.losses.log_loss(labels=self.y, predictions=self.logits)

            #Perform a softmax to a weights mean squared sum .
            elif(self.config["loss"] == "weighted_squares"):
                self.y = tf.placeholder(dtype=tf.int32, shape=(None), name="y")
                self.logits = tf.nn.softmax(z4)

                #prediction equals the largest value of the softmax
                self.predictions = self.logits[:, 0] < self.logits[:, 1]

                y = tf.one_hot(indices=tf.squeeze(self.y), depth=2)
                square_diff = tf.square(y - self.logits)
                good_cred, bad_cred = tf.split(value=square_diff, num_or_size_splits=2, axis=1)
                self.loss = self.config["false_neg_loss"]*tf.reduce_sum(good_cred) + self.config["false_pos_loss"]*tf.reduce_sum(bad_cred)

            #Perfrom a softmax to cross entropy
            else:
                self.y = tf.placeholder(dtype=tf.int32, shape=(None, self.config["outputs"]), name="y")
                y = tf.squeeze(self.y)
                self.logits = tf.nn.softmax(z4)
                _, self.predictions = tf.nn.top_k(self.logits)
                self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z4)
                self.loss = tf.reduce_mean(self.xentropy, name="loss")

        with tf.name_scope("lipshitz_regularization"):
            #
            self.w1_projection, self.w1_norm = tf_helper.lipschitz_projection(self.w1, p_norm=self.config["p_norm"], lamda=self.config["lamda1"])
            self.w2_projection, self.w2_norm = tf_helper.lipschitz_projection(self.w2, p_norm=self.config["p_norm"], lamda=self.config["lamda2"])
            self.w3_projection, self.w3_norm = tf_helper.lipschitz_projection(self.w3, p_norm=self.config["p_norm"], lamda=self.config["lamda3"])

        with tf.name_scope("training_op"):
            self.learning_rate = tf.Variable(self.config["learning_rate"])
            #optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.config["momentum"], use_nesterov=True)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            #computes gradients and applys gradients
            self.training_op = optimizer.minimize(self.loss)

        self._build_metrics()

    def _build_metrics(self):

        with tf.name_scope("accuracy"):
            self.train_acc, self.train_acc_op = tf.metrics.accuracy(labels=self.y, predictions=self.predictions, name="train_acc")
            self.test_acc, self.test_acc_op = tf.metrics.accuracy(labels=self.y, predictions=self.predictions, name="test_acc")

            #keeps a handle on local accuracy variables
            self.train_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy/train_acc")
            self.test_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy/test_acc")
            self.train_acc_vars_init = tf.variables_initializer(var_list=self.train_acc_vars)
            self.test_acc_vars_init = tf.variables_initializer(var_list=self.test_acc_vars)


        with tf.name_scope("precision"):
            self.train_prec, self.train_prec_op = tf.metrics.precision(labels=self.y, predictions=self.predictions, name="train_prec")
            self.test_prec, self.test_prec_op = tf.metrics.precision(labels=self.y, predictions=self.predictions, name="test_prec")

            #keeps a handle on local precision variables
            self.train_prec_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision/train_prec")
            self.test_prec_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision/test_prec")
            self.train_prec_vars_init = tf.variables_initializer(var_list=self.train_prec_vars)
            self.test_prec_vars_init = tf.variables_initializer(var_list=self.test_prec_vars)

        with tf.name_scope("recall"):
            self.train_rec, self.train_rec_op = tf.metrics.recall(labels=self.y, predictions=self.predictions, name="train_rec")
            self.test_rec, self.test_rec_op = tf.metrics.recall(labels=self.y, predictions=self.predictions, name="test_rec")

            #keeps a handle on local precision variables
            self.train_rec_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall/train_rec")
            self.test_rec_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall/test_rec")
            self.train_rec_vars_init = tf.variables_initializer(var_list=self.train_rec_vars)
            self.test_rec_vars_init = tf.variables_initializer(var_list=self.test_rec_vars)

        with tf.name_scope("cost_matrix"):
            self.cost_matrix = [tf.count_nonzero(self.y < tf.cast(self.predictions, tf.int32))]
    def save_model(self, session, save_dir, file_name="model.ckpt"):
        self.saver.save(session, os.path.join(save_dir, file_name))
