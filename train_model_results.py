#Change Point DetectionTrace

#from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
#from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.attributes import attributes_filter
from numpy import array
import numpy as np
from sklearn.preprocessing import LabelEncoder

import ruptures as rpt
from ruptures.metrics import precision_recall
from ruptures.metrics import hausdorff
from ruptures.metrics import randindex


import matplotlib.pyplot as plt
import pm4py
from sklearn.utils import shuffle
from itertools import cycle
from ruptures.utils import pairwise

from pm4py.objects.log.importer.xes import importer as xes_importer
from datetime import datetime, timezone
import os


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS


from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
import json
#import pyodbc



from scipy.spatial import distance as sci_distance
from sklearn import cluster as sk_cluster
from scipy import stats
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification

from kneed import KneeLocator



from sklearn.neighbors import NearestNeighbors

import seaborn as sns

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding


from keras.callbacks import EarlyStopping

from keras.optimizers import Adam
from keras.metrics import categorical_accuracy

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import seaborn as sns


n_input = 2
n_output = 1
LSTM_CELLS = 50
len_of_points = 1
index = 0 #fileinfo index
resample_dataset = False
plt.rcParams['font.size'] = '12'


def lstm_algorithm(index, len_of_points, resample_dataset, LSTM_CELLS):
    
    #print('Current path: ',os.getcwd())
    if not os.path.exists('results'):
        os.makedirs('results')
    
    
    
    COLOR_CYCLE = ["#4286f4", "#f44174"]
    
    split_percentage = 0.8
                   
    """                        
    answer = input('Give me the length of window: ')
    if answer == 'max':
        len_of_points = 0
    else:
        len_of_points = int(answer)
       
    #len_of_points = 3
    """
    
    
    
    fileinfo = {
        0: {'filename': 'base_kasteren.csv', 'separator': ' ', 'columns': ['date','time','attr1','attr2','state','concept:name']},
        1: {'filename': 'activity3.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},
        2: {'filename': 'activitylog_uci_detailed_labour.xes', 'separator': '', 'columns': []},
        3: {'filename': 'atmo1.csv', 'separator': ' ', 'columns': ['date','time','concept:name','state','activity']},        
        4: {'filename': 'activity1.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},
        5: {'filename': 'activity2.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},
        6: {'filename': 'espa.xes', 'separator': ';', 'columns': []},    
        7: {'filename': 'activity3.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},
        8: {'filename': 'activity4.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},
        9: {'filename': 'activity5.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},
        10: {'filename': 'activity6.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},        
        11: {'filename': 'activity7.csv', 'separator': ',', 'columns': ['id','case:concept:name','subjectID','attr_starttime','time:timestamp','concept:name','label_subactivity']},        
        12: {'filename': 'BPI_Challenge_2017.xes', 'separator': '', 'columns': []},
        
        
        
        
        }
    
    #choose file 
    
    filename = fileinfo[index]['filename']
    filepath = '../datasets/' + fileinfo[index]['filename']
    
    dataframe = pd.DataFrame() 
    if not os.path.exists('results/'+filename):
        os.makedirs('results/'+filename)
    if not os.path.exists('results/'+filename+'/'+ str(len_of_points) + '/'):
        os.makedirs('results/'+filename+'/'+str(len_of_points)+'/')
        
    #if it is a csv file    
    if (filename.find('.csv') != -1): 
        #load file to dataframe
        dataframe = pd.read_csv (filepath, sep = fileinfo[index]['separator'], names = fileinfo[index]['columns'],low_memory=False)
        #for Kastern dataset prepare columns
        if index in [0,3,12]:
            dataframe['time:timestamp'] = dataframe['date'] + ' ' + dataframe['time']
            dataframe['case:concept:name'] = dataframe['date']
            #dataframe = dataframe[dataframe['concept:name']!='None']  
               
        #print ("file is csv ") 
        #print(dataframe.head(20))
        
        #drop nan
        
        #convert csv to xes
        log = pm4py.convert_to_event_log(dataframe)
    
    else:
        #the file is xes
        #import log
        #xes_importer.iterparse.Parameters.MAX_TRACES = 10
        #parameters = {xes_importer.iterparse.Parameters.MAX_TRACES: 50}
        #log = xes_importer.apply('datasets/BPI Challenge 2018.xes.gz', parameters=parameters)
        log = pm4py.read_xes(filepath)
        print(log)
        #convert to dataframe
        dataframe = pm4py.convert_to_dataframe(log)
        print(dataframe)
        #print(dataframe['time:timestamp'][0].replace(tzinfo=timezone.utc).astimezone(tz=None))
        #dataframe['time:timestamp'] = dataframe['time:timestamp'].dt.tz_convert(None)
    
    if index in [2,12]:
        #process time:timestamp remove zone information
        dataframe['time:timestamp'] = dataframe['time:timestamp'].dt.tz_convert(None) 

        
    
    #del log
    print('Dataframe print\n',dataframe)
    #get only start events if lifecycle:transition if column does not exists create it
    if 'lifecycle:transition' in dataframe.columns:  
        
        dataframe = dataframe[dataframe['lifecycle:transition'] == 'complete']
        
    else:    
        dataframe['lifecycle:transition'] = 'complete';
        
    
    #remove Start and End events
    dataframe = dataframe[dataframe['concept:name']!='Start']    
    dataframe = dataframe[dataframe['concept:name']!='End']  
    
    #sort by time
    if 'time:timestamp' in dataframe.columns:
        dataframe = dataframe.sort_values('time:timestamp')
    else:
        print('Error: no column time:timestamp in event log')
    
    #print('Sorted dataframe\n',dataframe)
    
    
    #plot time vs activity
    #fig, axes = plt.subplots(1, 1, figsize=(100, 100))
    #fig = dataframe.plot(x='time:timestamp', y='concept:name', kind="scatter").get_figure()
    #fig.savefig('results/'+filename+'/'+str(len_of_points) +'/conceptname.png', bbox_inches='tight')
    
    #plot time vs trace id
    #df.plot(x='col_name_1', y='col_name_2', style='o')
    #fig = dataframe.plot(x='time:timestamp', y='case:concept:name', kind="scatter").get_figure()
    #fig.savefig('results/'+filename+'/'+str(len_of_points) +'/caseconceptname.png', bbox_inches='tight')
    
    #keep only mandatory columns
    dataframe = dataframe[['case:concept:name','concept:name','time:timestamp']]
    #convert sorted dataframe to log
    log = pm4py.convert_to_event_log(dataframe)
    
    #initial_df = dataframe.copy()
    #print('Initial dataframe\n',initial_df)
    #-----------------------------------------------------------------
    ############################################################
    #-------------- Resample -----------------------------------
    ###########################################################
    if resample_dataset:
        #preprocess timestamp to be prepared for resample
        #make time:timestamp datetime
        dataframe.loc[:,'time:timestamp'] = pd.to_datetime(dataframe['time:timestamp'])
        #set time:timestamp as index
        dataframe = dataframe.set_index(["time:timestamp"])
        #remove duplicates
        #print('Duplicated\n')
        #print(dataframe[dataframe.index.duplicated()])
        dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
        
        
        #------Reasample dataframe every 5min keep last value found if Nan-------------
        dataframe = dataframe.resample("5T").fillna("backfill")    
        print('Resample',dataframe)
        #print( dataframe.last())
    
    #save resampled dataframe to csv
    dataframe.to_csv('../datasets/resampled_sorted_df.csv')
    
    #dataframe is initial event log sorted by time (start event only)
    #convert sorted by time dataframe back to log (xes)
    #log = pm4py.convert_to_event_log(dataframe)
    
    #-----------------------------------save to csv-------------------------------------
    #uncomment only if you need it
    #dataframe.to_csv('datasets/activitylog_uci_detailed_labour.csv')
    
    #print('\nDataframe LOG\n',dataframe)
    
    #--------------- Concat activities of a trace in one row ---------------
    #concat events with same case:concept:name (space separated)     
    print('dataframe\n',dataframe)
    df = dataframe.groupby('case:concept:name', sort = False).agg({'concept:name': lambda x: ' '.join(x)})
    print('df\n',df)
    df = df.reset_index()
    if len_of_points:
        print('--------------------------------------')
        df['concept:name'] = df['concept:name'].apply(lambda x: list(x.split(' ')))
        df['concept:name'] = df['concept:name'].apply(lambda x: [x[i:i+len_of_points] for i in range(0, len(x), len_of_points)])
        df = df.set_index('case:concept:name')['concept:name'].apply(pd.Series).stack().reset_index(level=0).rename(columns={0:'concept:name'})
        df['concept:name'] = df['concept:name'].apply(lambda x: ' '.join(x))
        print('\ndftest\n',df)
    
    df = df.reset_index() #check here
    
    #del dataframe
    
    
    #print the activities of the log
    activities = pm4py.get_attribute_values(log,'concept:name')
    print('\nActivities:\n',activities)
    
    #split data from event log - 80% for train and 20% for test
    
    #shuffle before split
    #df = shuffle(df)
    #print('df', df,'\n')
    
    
    #----------------------- Split Train and Test data ----------------------
    #split rows depending on percentage
    split_rows = int(df.shape[0]*split_percentage)
    print('Split Rows', split_rows,'\n')
    
    #train dataframe
    train_df = df[:split_rows]
    train_df.to_csv('train.csv')
    print('Train Rows', train_df,'\n')
    
    #test dataframe
    test_df = df[split_rows:]
    test_df.to_csv('test.csv')
    #print('Test Rows', test_df,'\n')
    
    
    # --------------------------------------------------------------
    
    #data = df['concept:name'].copy().to_list()
    data = train_df['concept:name'].copy().to_list()
    #Just for Eating/Drinking
    #data = data.replace('Eating/Drinking','EatDrink')
    #print('Data\n',data)
    
    tokenizer = Tokenizer()
    #reads the words in data and gives an index for every words based on frequency
    tokenizer.fit_on_texts([data])
    print('Word index: ')
    print(tokenizer.word_index)
    
    #replace every word in the text to correspoding word index - returns list of list with one element so use [0] to get the one and only first list
    encoded = tokenizer.texts_to_sequences([data])[0]
    #print('encoded: \n')
    #print(encoded)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    #print('list\n',[e for e in encoded])
    #print('Min ',min([len(e) for e in encoded]))
    
    # LSTM 3 timesteps - prepare data - encode 2 words -> 1 word
    sequences = list()
    for i in range(n_input, len(encoded)):
    	sequence = encoded[i-n_input:i+1]
    	sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))
    print('Sequences: \n')
    print(sequences)
    
    max_length = max([len(seq) for seq in sequences])    #max_length is 3
    # Pad sequence to be of the same length
    # length of sequence must be 3 (maximum)
    # 'pre' or 'post': pad either before or after each sequence
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    print('Max Sequence Length: %d' % max_length)   													
    
    #convert list to array to get X,y train
    sequences = array(sequences)
    X, y = sequences[:,:-1],sequences[:,-1]
    print('X: \n')
    print(X)
    print('y: \n')
    print(y)
    
    #convert y to binary vectors
    y = to_categorical(y, num_classes=vocab_size)
    print('y: \n')
    print(y)
    
    #test data
    test_data = test_df['concept:name'].copy().to_list()
    
    
    test_encoded = tokenizer.texts_to_sequences([test_data])[0]
    
    test_sequences = list()
    
    for i in range(n_input, len(test_encoded)):
    	test_sequence = test_encoded[i-n_input:i+1]
    	test_sequences.append(test_sequence)
    max_length = max([len(seq) for seq in test_sequences])
    test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='pre')
    
    test_sequences = array(test_sequences)
    test_X, test_y = test_sequences[:,:-1],test_sequences[:,-1]
    
    #convert y to binary vectors
    test_yl = to_categorical(test_y, num_classes=vocab_size)
    
    
    
    
    
    model = Sequential()
    #the first layer
    # - the largest integer (i.e. word index) in the input should be no larger than vocabulary size 
    # - The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset.
    # - output_dim (50): This is the size of the vector space in which words will be embedded (size of the embedding vectors). It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
    # - input_length: This is the length of input sequences (here is 2)
    # The Embedding layer has weights that are learned. If you save your model to file, this will include weights for the Embedding layer.
    # The output of the Embedding layer is a 2D vector with one embedding for each word in the input sequence of words (input document).
    # If you wish to connect a Dense layer directly to an Embedding layer, you must first flatten the 2D output matrix to a 1D vector using the Flatten layer.
    
    model.add(Embedding(vocab_size+1, LSTM_CELLS, input_length=max_length-1))
    
    model.add(LSTM(vocab_size))
    model.add(Dropout(0.1))
    model.add(Dense(vocab_size, activation='softmax'))
    opt = Adam(learning_rate=0.001)                                                                                                                 
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X, y, epochs=500, verbose=0,batch_size = 20,validation_data=(test_X, test_yl))
    
    print(model.summary())
    model.save('lstm_model.h5')  # creates a HDF5 file 
    #del model  # deletes the existing model
    

    
    #predict sequence of n_words activities
    def generate_seq(model, tokenizer, max_length, seed_text, n_words):
      #get input activity
      in_text = seed_text
      #print('in_text',in_text,'\n')
      #for the number of activities on sequence you want to predict
      for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        #pad if less than max text length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        #print('in text ',in_text)
        #predict one activity
        #yhat = model.predict_classes(encoded, verbose=0)
        yhat = np.argmax(model.predict(encoded), axis=-1)
        out_word = ''
        for word, index in tokenizer.word_index.items():
          #convert predicted activity to word
          if index == yhat:
            #print('Word',word,'\n')
            out_word = word
            break
        #feed the next input with the sequence of activities
        in_text += ' ' + out_word
        
      return in_text
     
    
    
    #load trained model
    #model = load_model('lstm_model.h5')
    
    
    
    # Evaluate network
    print('LSTM Network Evaluation:\n')
    train_score = model.evaluate(X, y, verbose=0)
    print('Train Score\n',train_score)
    score = model.evaluate(test_X, test_yl, verbose=0)
    print('Test Score\n')
    print(score)
    
    print('History\n')
    print(history.history.keys())
    # plot loss during training
    fig = plt.figure()
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    fig.savefig('results/'+filename+'/'+str(len_of_points) +'/Loss.png', bbox_inches='tight')
    # plot accuracy during training
    fig = plt.figure()
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
    fig.savefig('results/'+filename+'/'+str(len_of_points) +'/Accuracy.png', bbox_inches='tight')
    
    
    
    
    
    
    
    
    print('LSTM Results: ')
    print ('\n')
    #generated_text = ''
    #sequence prediction
    for i in tokenizer.word_index:
      #print(tokenizer.index_word)
      w = generate_seq(model, tokenizer, max_length-1, i , n_input+1)
      #generated_text = generated_text.join('\n'+w)
      print(w)  
      
    print('LSTM Results: ')
    print ('\n') 
    #for i in tokenizer.word_index:
    #	print(generate_seq(model, tokenizer, max_length-1, i , 1))
    all_data = df['concept:name'].copy().to_list()
    
    all_encoded = tokenizer.texts_to_sequences([all_data])[0]
    
    all_sequences = list()
    
    for i in range(n_input, len(all_encoded)):
    	all_sequence = all_encoded[i-n_input:i+1]
    	all_sequences.append(all_sequence)
    max_length = max([len(seq) for seq in all_sequences])
    all_sequences = pad_sequences(all_sequences, maxlen=max_length, padding='pre')
    
    all_sequences = array(all_sequences)
    all_X, all_y = all_sequences[:,:-1],all_sequences[:,-1]
    
    #convert y to binary vectors
    all_yl = to_categorical(all_y, num_classes=vocab_size)
    
    #load trained model
    #model = load_model('lstm_model.h5')
    
    #print('Tokenizer \n',tokenizer)
    print('Tokenizer word index\n',tokenizer.word_index)
    
    np.set_printoptions(suppress=True)
    cnt = 0
    for i in range(len(all_X)):
      #yhat = model.predict_classes(all_X[i].reshape(1,2,1), verbose=0)
      yhat = np.argmax(model.predict(all_X[i].reshape(1,n_input,1)), axis=-1)
      df.loc[i,'X_input'] = str(all_X[i])
      df.loc[i,'Expected'] = all_y[i]
      df.loc[i,'predicted'] = yhat
      
      #print('Expected:', all_y[i] , 'Predicted', yhat)
      prob = model.predict_proba(all_X[i].reshape(1,n_input,1))[0]
      df.loc[i,'probabilities'] = ' '.join([str(elem) for elem in list(prob)]) 
      if (all_y[i] == yhat):
        df.loc[i,'result'] = 'ok'  
        cnt += 1
      else:  
        df.loc[i,'result'] = 'Error'  
    
    #print(df['predicted'].replace(tokenizer.word_index))      
    df.to_csv('results/'+filename+'/'+str(len_of_points) +'/resample_'+str(resample_dataset)+'_lstm.csv')
    print('Total successful: ',cnt,' out of ', len(all_X), 'Percentage: ', cnt/len(all_X))
    
    
    
    
    
     
    # predict probabilities for test set
    yhat_probs = model.predict(test_X, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(test_X, verbose=0)
    print('yhat_classes\n',yhat_classes)
    # reduce to 1d array
    #yhat_probs = yhat_probs[:, 0]
    #yhat_classes = yhat_classes[:, 0]
     
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(test_y, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(test_y, yhat_classes,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(test_y, yhat_classes,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(test_y, yhat_classes,average='weighted')
    print('F1 score: %f' % f1)
     
    # kappa
    kappa = cohen_kappa_score(test_y, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    #auc = roc_auc_score(test_y, yhat_probs,multi_class='ovr')
    #print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(test_y, yhat_classes)
    print(matrix)  
    fig = plt.figure()
    sns.heatmap(matrix,center=True)
    plt.show()
    fig.savefig('results/'+filename+'/'+str(len_of_points) +'/ConfusionMatrix.png', bbox_inches='tight')
    
    
    #headers
    #filename - resample - len of points - train loss + Accuracy - test score
    #write results to csv
    fd = open("total_results.csv", "a+")
    row = filename + '\t' + str(resample_dataset)+ '\t' + str(len_of_points) + '\t' + str(train_score[0])+ '\t' + str(train_score[1]) + '\t' + str(score[0])+ '\t' + str(score[1]) + '\t' + str(accuracy) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(kappa) + '\t'+ '' + '\t' + json.dumps(tokenizer.word_index) + '\n'
    fd.write(row)
    fd.close()


for i in [0,1,2,4,5,6,7,8,9,10,11,12]:
    lstm_algorithm(i, 1, False, 50)
  