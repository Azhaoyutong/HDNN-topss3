from tensorflow.python.keras import optimizers
import tensorflow  as tf
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef
import tensorflow as tf
from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
from keras.callbacks import Callback
from keras.initializers import Constant
from keras import backend as K
K.clear_session()
import matplotlib.pyplot as plt
import numpy as np
import argparse

mdel = open("./model/model.txt", "r")
state = mdel.readline().strip().split("   ")
print(state)
ostate = mdel.readline().strip().split("   ")
print(ostate)
pstate = mdel.readline().strip().split("   ")
print(pstate)
# osym="MmNHhGCEeF"
psym = ["M", "H", "C", "E", "B", "D"]
osym_dict = {"M": 0, "m": 1, "N": 2, "H": 3, "h": 4, "G": 5, "C": 6, "E": 7, "e": 8, "F": 9, "B": 10, "D": 11}
psym_dict = {"M": 0, "H": 1, "C": 2, "E": 3, "B": 4, "D": 5}
ESYM = "ACDEFGHIKLMNPQRSTVWY"
window_length = 21
nb_lstm_outputs = 10
rows, cols = window_length, 52
nb_time_steps = window_length
from keras import backend as K

# my_loss
def my_loss(y_true, y_pred):
    return K.mean(tf.math.log(y_pred) * y_true, axis=-1)

# viterbi
def compute(A, Obser, count, y_pred):
    #   max_p
    N = 12
    max_p = np.zeros((len(Obser), N))
    #   path
    path = np.zeros((N, len(Obser)))
    #   initialization
    for i in range(N):
        if i == 0 or i == N - 1:
            e = 0
        else:
            e = y_pred[osym_dict[ostate[i]]][count][0][0]
        max_p[0][i] = e * A[0][i]
        path[i][0] = i
    for t in range(1, len(Obser)):
        newpath = np.zeros((N, len(Obser)))
        for y in range(N):
            prob = -1
            if y == 0 or y == N - 1:
                e = 0
            else:
                e = y_pred[osym_dict[ostate[y]]][count][t][0]
            for y0 in range(1, N):
                nprob = max_p[t - 1][y0] * A[y0][y] * e
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
    for y in range(1, N):
        if max_p[len(Obser) - 1][y] > max_prob:
            max_prob = max_p[len(Obser) - 1][y]
            path_state = y
    return path[path_state]


def evaluation(y_predscore, A,x_fasta_path,y_out_path):
    trainsion_probility = A
    pre = []
    lab = []
    with open(y_out_path+"result.txt", 'w') as fw:
        f = open(x_fasta_path, "r")
        l = f.readline()
        count = 0
        while l:
            if l[0] == ">":
                pdb_id = l[0:7]
                print(pdb_id)
                fw.write(pdb_id + "\n")
            if l[0] != ">" and l != "\n":
                obs_seq = l.strip()
                path = compute(trainsion_probility, obs_seq, count, y_predscore)
                count = count + 1
                pred = []
                for i in path:
                    pred.append(psym_dict[pstate[int(i)]])
                    pre.append(psym_dict[pstate[int(i)]])
                    fw.write(pstate[int(i)])
                fw.write("\n")
            l = f.readline()

def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(nb_time_steps, activation='relu', name='attention_dense')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def predict_iter1(A,x_fasta,x_inputfeature,outpath):
    valid_class = []
    # from keras.models import Sequential
    for o_v in "MmNHhGCEeF":
        for k in range(12):
            if ostate[k] == o_v:  # Judge whether the shared state label corresponding to the state is consistent with all the neural network classifications
                print("label:", o_v)
                #f = open("./datasets/test.txt", "r")
                x_fasta_path=x_fasta
                f = open(x_fasta_path, "r")
                if o_v.islower() == True:
                    o_v = o_v + "_1"
                y_valid_predict = []
                # model
                input3 = Input(shape=(window_length, 52), name='input3')
                re = Reshape((window_length, 52, 1))(input3)
                conv1 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(re)
                conv1 = BatchNormalization(axis=-1)(conv1)
                conv1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(conv1)
                out2 = Reshape((window_length, -1))(conv1)
                lstm_out = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm1')(out2)
                lstm_out2 = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm2')(lstm_out)
                attention_mul = attention_3d_block(lstm_out2)
                attention_flatten = Flatten()(attention_mul)
                drop2 = Dropout(0.3)(attention_flatten)
                fc1 = Dense(50, activation='relu', kernel_initializer='random_uniform',
                            bias_initializer='zeros')(drop2)
                fc2 = Dense(10, activation='relu')(fc1)
                output1 = Dense(1, activation='sigmoid', name='output_1')(fc2)
                model = Model(inputs=input3, outputs=output1)
                #model = load_model("C:/Users/ZYT/Desktop/CHMM_DNN4class/cochdnn-chdnn/models12_easy/" + o_v + "/trained_model.h5")
                model.summary()
                model.load_weights("./model/dnn_model_12/" + o_v + "_my_model_weights.h5")
                #x_valid = np.load("C:/Users/ZYT/Desktop/CHMM_DNN4class/x_test_new_hhm31_winlen_21.npy", allow_pickle=True)

                for s in range(len(x_inputfeature)):
                    a = np.array(x_inputfeature[s])
                    y_valid_pred = model.predict(a, batch_size=128)
                    y_valid_predict.append(y_valid_pred)
                valid_class.append(y_valid_predict)
    evaluation(valid_class, A,x_fasta_path,outpath)
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
    parser.add_argument('-o', '--output_path', default='./result/')
    args = parser.parse_args()
    # generate data
    from pre_processing_oneseq import Processor
    fasta = args.fasta
    hhblits_path = args.hhblits_path
    output_path = args.output_path
    processor = Processor()
    x_test = processor.data_pre_processing(fasta, hhblits_path, window_length)
    A = np.load("./model/A.npy")
    predict_iter1(A,fasta,x_test,output_path)
    print("The prediction task using Co-HDNNtopss has been completed, and the results have been recorded in the given path.")

