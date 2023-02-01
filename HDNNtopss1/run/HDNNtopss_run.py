#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import *
K.clear_session()
K.set_image_data_format('channels_last')
np.random.seed(0)
from keras.models import *
from keras.callbacks import Callback
from keras.initializers import Constant
from sklearn import model_selection
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import callbacks
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras
import time
from keras import Input
from keras.layers import Activation
from keras.layers import *
import argparse
import time

mdel=open("./model/model.txt","r")
state=mdel.readline().strip().split(" ")
ostate=mdel.readline().strip().split("   ")
pstate=mdel.readline().strip().split("   ")
osym="MmNHhGCEeF"
psym=["M","H","C","E","B","D"]
osym_dict= {"M": 0,"m": 1,"N": 2,"H": 3,"h": 4,"G":5,"C":6,"E":7,"e":8,"F":9,"B":10,"D":11}
psym_dict= {"M":0,"H":1,"C":2,"E":3,"B":4,"D":5}
deep_dict={"H":0,"E":1,"C":2,"M":3,"B":4,"D":5}
deep_con_dict={0:"M",1:"H",2:"C",3:"E"}
K.set_image_data_format('channels_last')
np.random.seed(0)
# one-hot map
dict_AA = {'C': 0, 'D': 1, 'S': 2, 'Q': 3, 'K': 4,
        'I': 5, 'P': 6, 'T': 7, 'F': 8, 'N': 9,
        'G': 10, 'H': 11, 'L': 12, 'R': 13, 'W': 14,
        'A': 15, 'V': 16, 'E': 17, 'Y': 18, 'M': 19}
# attention mechanism
def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(nb_time_steps, activation='relu', name='attention_dense')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# viterbi
def compute(A, Obser,count,y_pred):
    #   max_p
    N=12
    max_p = np.zeros((len(Obser), N))
    #   path
    path = np.zeros((N, len(Obser)))
    #   initialization
    for i in range(N):
        if i == 0 or i == N-1:
            e = 0
        else:
            e = y_pred[count][0][psym_dict[pstate[i]]]
        max_p[0][i]=e *A[0][i]
        path[i][0] = i
    for t in range(1, len(Obser)):
        newpath = np.zeros((N, len(Obser)))
        for y in range(N):
            prob = -1
            if y == 0 or y == N-1:
                e = 0
            else:
                e = y_pred[count][t][psym_dict[pstate[y]]]
            for y0 in range(1,N):
                nprob = max_p[t-1][y0] * A[y0][y] * e
                if nprob > prob:
                    prob = nprob
                    state = y0
                    #   record path
                    max_p[t][y] = prob
                    for m in range(t):
                        newpath[y][m] = path[state][m]
                    newpath[y][t] = y
        path = newpath
    max_prob = -1
    path_state = 1
    #   returns the path with the maximum probability
    for y in range(1,N):
        if max_p[len(Obser)-1][y] > max_prob:
            max_prob = max_p[len(Obser)-1][y]
            path_state = y
    return path[path_state]

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

    '''
    cmd = python run.py --fasta ../datasets/test.txt --hhblits_path ../datasets/test_hmm/ --output_path ../result
    cmd = python run.py -f ../datasets/test.txt -p ../datasets/test_hmm/ -o ../result
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta',
                        default="./datasets/test.txt")
    parser.add_argument('-p', '--hhblits_path',
                        default="./datasets/test_hhm/")
    parser.add_argument('-o', '--output_path', default='./result')
    args = parser.parse_args()
    # calculate running time
    time_start = time.time()
    # generate data
    from pre_processing import Processor
    window_length = 19
    nb_lstm_outputs = 100
    rows, cols = window_length, 52
    nb_time_steps = window_length
    processor = Processor()
    fasta = args.fasta
    hhblits_path = args.hhblits_path
    output_path = args.output_path
    x_test = processor.data_pre_processing(fasta, hhblits_path, window_length)

    # model
    input1 = Input(shape=(window_length, 21), name='input1')
    re = Reshape((window_length, 21, 1))(input1)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(re)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(conv1)
    conv1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv1)
    out1 = Reshape((window_length, -1))(conv1)
    input2 = Input(shape=(window_length, 31), name='input2')
    re = Reshape((window_length, 31, 1))(input2)
    conv1 = Conv2D(filters=15, kernel_size=3, strides=1, padding='same', activation='relu')(re)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(conv1)
    conv1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv1)
    conv1 = Conv2D(filters=64, kernel_size=7, strides=1, padding='same', activation='relu')(conv1)
    conv1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv1)
    out2 = Reshape((window_length, -1))(conv1)
    merged = concatenate([out1, out2], axis=-1)
    lstm_out = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm1')(merged)
    lstm_out2 = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm2')(lstm_out)
    attention_mul = attention_3d_block(lstm_out2)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.25)(attention_flatten)
    fc1 = Dense(700, activation='relu', kernel_initializer='random_uniform',
                bias_initializer='zeros')(drop2)
    fc2 = Dense(100, activation='relu')(fc1)
    output1 = Dense(4, activation='softmax', name='output_1')(fc2)
    model = Model(inputs=[input1, input2], outputs=output1)
    #model=load_model("C:/Users/ZYT/Desktop/CHMM_DNN4class/cochdnn-chdnn/chdnn分开训练_TMPSS_code_model/models_4/trained_model.h5")
    model.summary()

    # load weights
    model.load_weights("./model/trained_weights.h5")

    # nn predict score as emission of hmm
    y_pred1 = model.predict([x_test[:, :, 0:21], x_test[:, :, 21:52]], batch_size=64)
    time_end = time.time()
    Y_pred = np.argmax(y_pred1, axis=-1)
    topo_dict = {"M":0,"H":1,"C":2,"E":3,"B":4,"D":5}
    with open(fasta) as get_fasta:
        score_dataset = []
        Y_pred_dataset=[]
        Y_pred_allseq=[]
        temp = get_fasta.readline()
        pdb_id = ""
        index = 0
        while temp:
            if (temp[0] == ">"):
                pdb_id = temp[1:].strip()
                temp = get_fasta.readline()
                continue
            score_line = []
            Y_pred_line=[]
            for i in temp:
                if (i != '\n'):
                    score_line.append(y_pred1[index])
                    Y_pred_line.append(psym_dict[deep_con_dict[Y_pred[index]]])
                    Y_pred_allseq.append(psym_dict[deep_con_dict[Y_pred[index]]])
                    index += 1
            score_dataset.append(score_line)
            Y_pred_dataset.append(Y_pred_line)
            temp = get_fasta.readline()
        y_predscore = score_dataset
        Y_preddata=Y_pred_dataset
    # viterbi decoding and result
    invisiable = {"M":0,"H":1,"C":2,"E":3,"B":4,"D":5}
    topo_ss="MHCE"
    invisiable_ls = [0, 1, 2, 3]
    trainsion_probility = np.load("./model/A.npy")
    fw = open(args.output_path+"/result.txt", "w")
    f = open(fasta, "r")
    l = f.readline()
    obs_seq = ""
    count = 0
    while l:
        if l[0]==">":
            pdb_id=l[0:7]
            print(pdb_id)
        if l[0] != ">" and l != "\n":
            obs_seq = l.strip()
            fw.write(pdb_id + "|seq_len" + str(len(obs_seq)) + "\n")
            path = compute(trainsion_probility, obs_seq, count,y_predscore)
            count = count + 1
            fresult = []
            pred = []
            for i in path:
                fw.write(pstate[int(i)])
            fw.write("\n")
        l = f.readline()
    print("The prediction task using HDNNtopss has been completed, and the results have been recorded in the given path.")





