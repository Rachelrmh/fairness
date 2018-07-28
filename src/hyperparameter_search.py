import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing as pp
from SCHUFAModel import SCHUFAModel
from datetime import datetime
import plots
import os

def train_model(train_data, train_labels, test_data, test_labels, config, lipschitz_constraint=False):
    '''
    Train custon DNN model with SCHUFA dataset

    Parameters:
        train_data, train_labels, test_data, test_labels: Numpy matrices containing data
        config: A dictionary containing configuration parameters
        lipschitz_constraint: A boolean if lipschitz regularization should be enforced or not
    Returns:
        metrics_data: A dictionary with evaluation metrics for the training and testing set
    '''

    num_batches = train_data.shape[0]//config["batch_size"]

    metrics_data = {"w1_norm":[], "w2_norm":[], "w3_norm":[], "train_loss":[], "test_loss":[],
                    "train_acc":[], "train_prec":[], "train_rec":[], "test_acc":[], "test_prec":[],
                    "test_rec":[]}

    tf.reset_default_graph()

    #Initialize SCHUFA model
    model = SCHUFAModel(config)
    test_feed_dict = {model.x:test_data, model.y:test_labels}
    print("Starting Training")
    start = datetime.now()

    global_var_init = tf.global_variables_initializer()

    with tf.Session() as sess:

        #Initialize local accuracy metrics variables
        sess.run([model.train_acc_vars_init, model.test_acc_vars_init])
        #Intializer local precision metrics variables
        sess.run([model.train_prec_vars_init, model.test_prec_vars_init])
        #Initialize local recall metrics variables
        sess.run([model.train_rec_vars_init, model.test_rec_vars_init])

        global_var_init.run()

        for epoch in range(config["num_epochs"]):
            #reset local updating variables for training evaluation metrics
            sess.run([model.train_acc_vars_init, model.train_prec_vars_init, model.train_rec_vars_init])
            sess.run([model.test_acc_vars_init, model.test_prec_vars_init, model.test_rec_vars_init])
            train_loss_accum = 0

            for batch in range(num_batches):
                #perform training operation
                train_batch_data, train_batch_labels = get_mini_batch(train_data, train_labels, batch, num_batches)
                feed_dict = {model.x:train_batch_data, model.y:train_batch_labels}
                sess.run(model.training_op, feed_dict=feed_dict)

                #compute mini-batch loss, accuracy, precision, and recall
                train_loss_accum += sess.run(model.loss, feed_dict=feed_dict)
                sess.run([model.train_acc_op, model.train_prec_op, model.train_rec_op], feed_dict=feed_dict)

                if(lipschitz_constraint):
                    #sess.run([model.w1_projection, model.w2_projection, model.w3_projection])
                    sess.run([model.w3_projection])
            #compute weight matrix norms after each full pass of data
            metrics_data["w1_norm"].append(sess.run(model.w1_norm))
            metrics_data["w2_norm"].append(sess.run(model.w2_norm))
            metrics_data["w3_norm"].append(sess.run(model.w3_norm))

            #Save metric evaluations
            metrics_data["train_loss"].append(train_loss_accum/num_batches)
            metrics_data["train_acc"].append(sess.run(model.train_acc))
            metrics_data["train_prec"].append(sess.run(model.train_prec))
            metrics_data["train_rec"].append(sess.run(model.train_rec))

            metrics_data["test_loss"].append(sess.run(model.loss, feed_dict=test_feed_dict))
            metrics_data["test_acc"].append(sess.run(model.test_acc_op, feed_dict=test_feed_dict))
            metrics_data["test_prec"].append(sess.run(model.test_prec_op, feed_dict=test_feed_dict))
            metrics_data["test_rec"].append(sess.run(model.test_rec_op, feed_dict=test_feed_dict))

            print("Training Progress {:2.1%}".format(float((epoch+1)/config["num_epochs"])), end="\r", flush=True)
        print("Training Progress {:2.1%}".format(float((epoch+1)/config["num_epochs"])))
        cost_matrix = sess.run(model.cost_matrix, feed_dict=test_feed_dict)
        print(cost_matrix)
        model.save_model(sess, config["model_dir"])
    del model
    print("Training Complete!", "Total Training Time:", datetime.now()-start)
    return(metrics_data)

def get_mini_batch(X, y, current_batch_num, batch_numbers):
    '''
    Get mini-batch of input data and labels. Each dataset will be split in n number of batches, and
    current batch number will select the necessary mini-batch from the subsets of the split.

    Parameters:
        X: Input dataset to be fed for training into the DNN
        y: The target dataset for the input data used to train the DNN on
        current_batch_num: The current iteration in training to select a new mini-batch.
        batch_numbers: The overall number of batchs to split the whole dataset into

    Returns:
        batch_X: The mini-batch for input training data
        batch_y: The mini-batch for target training data
    '''
    X, y= np.asarray(X), np.asarray(y)
    batch_X = np.split(X, batch_numbers)[current_batch_num]
    batch_y = np.split(y, batch_numbers)[current_batch_num]
    return batch_X, batch_y

def retrieve_preprocess_data(SCHUFA_DATA_FILE):
    '''
    Load up and preprocess data. All numberical attributes are centered at an average of zero
    and normalized. Categorical data is onehot encoded

    Parameters:
        SCHUFA_DATA_FILE: file containing the raw SCHUFA data
    Returns:
        input_data: Processed data to be fed to DNN
        labels: Labels corresponding to input_data
    '''
    #retrieve data
    raw_data = pd.read_csv(SCHUFA_DATA_FILE, sep=" ", header=None)
    data = np.array(raw_data) #convert data to numpy matrix

    #Normalize columns 1, 4, 12
    #normalized_data = data
    normalized_data = pp.mean_normalize_columns(data, [1, 4, 12])

    #Label encode and one hot encode columns 0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19
    proccessed_data = pp.label_encode(normalized_data, [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])
    input_data = pp.one_hot_encode(proccessed_data, [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])

    #split data and labels
    [input_data, labels] = np.split(input_data, [-1], axis=1)

    #Since labels are orginally 1 for bad and 2 for good, subtract 1 to get 0 for bad and 1 for good
    labels = np.reshape(np.subtract(labels, [1]), (-1, 1))

    return(input_data, labels)

def create_dir_if_needed(dir):
    '''
    If a directory does not exist, create it.
    '''
    if(not os.path.isdir(dir)):
        os.makedirs(dir)
        print("Created File Path", dir)

def main(test_num):
    '''
    Main is responsible for conduction the hyperparameter search for suitable lamdas and p_norms, as well as saving
    the data at each training instance

    Parameters:
        test_num: the current test being conducted
    '''
    SCHUFA_DATA_DIR = "C:\Machine_Learning\ML Projects\Fairness\\fairness\data\statlog_german.data.txt"
    WORKING_DIR = "C:\Machine_Learning\ML Projects\Fairness\\fairness"
    DATA_AQUISITION = os.path.join(WORKING_DIR, "data_aquisition")
    TEST_DIR = os.path.join(DATA_AQUISITION, "test_"+str(test_num))
    create_dir_if_needed(DATA_AQUISITION)
    create_dir_if_needed(TEST_DIR)

    config = {"inputs":61, "hidden1":100, "hidden2":100, "hidden3":100, "outputs":1, "p_norm":2, "lamda1":5, "lamda2":5, "lamda3":5,
                "learning_rate":0.001, "momentum":0.9, "batch_size":100, "num_epochs":2000, "dropout_prob":1.0, "loss":"log_loss",
                "false_neg_loss":1.0, "false_pos_loss":5.0}
    data, labels = retrieve_preprocess_data(SCHUFA_DATA_DIR)

    for i in range(1):

        dir = os.path.join(DATA_AQUISITION, "test_"+str(test_num), "train_unreg_"+str(i))
        create_dir_if_needed(dir)
        config["model_dir"] = dir

        train_data, train_labels, test_data, test_labels = pp.train_test_stratified_split(data, labels, test_size=0.2)
        metrics_data = train_model(train_data, train_labels, test_data, test_labels, config, lipschitz_constraint=True)
        plots.plot_metrics(metrics_data, config, display=False)

if __name__ == "__main__":
    test_num = 2
    main(test_num)
