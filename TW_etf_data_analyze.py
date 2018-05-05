import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator,FormatStrFormatter

stock_code = ['0050','0051','0052','0053','0054','0055','0056','0057','0058','0059',
'006201','006203','006204','006208','00690','00692','00701','00713']
def plot_data():
    None
def load_csv(csv_name,etf_number):
    with open(csv_name,'r') as csvfile :
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='"')
        x_axis = []
        y_axis = []
        for row in spamreader:
            if(etf_number in row[0]):
                y_axis.append(float(row[6]))
                time_tmp = row[1]
                x_axis.append(datetime.datetime.strptime(time_tmp,"%Y%m%d").date())
        print(len(y_axis))     
        print(len(x_axis))
        plt.figure(figsize=(16,9))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))
        plt.plot(x_axis[0:1200],y_axis[0:1200],'--')
        plt.xticks(rotation=45)
        plt.ylim(25,100)
        plt.show()

def calculation_averge(stock_price,avergedays):
    if(len(stock_price) > avergedays ):
        i_index = 1
        sum_tmp = 0
        while(i_index <= avergedays):
            sum_tmp += stock_price[-1*i_index]
            i_index +=1
        return (float(sum_tmp)/avergedays)
    else:
        return 0
def data_preprocesing(csv_name,etf_number):
    start_date_time = "20140101"
    with open(csv_name,'r') as csvfile :
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='"')
        x_axis = []
        y_axis = []
        data_index = 0
        feature_x_tmp = np.zeros(11)
        feature_y_tmp = np.zeros(5)+-1
        data_tmp = []
        save_x_flag = False
        save_y_flag = False
        save_allfeature = False
        clear_feature_tmp = False
        weekly_save_point = '0'
        weekly_save_tmp = '0'
        last_friday_price = 0
        last_stock_price = 0
        dict_save_x_feature_yearly = {}
        dict_save_y_feature_yearly = {}
        dict_save_x_feature_weekly = {}
        dict_save_y_result_weekly ={}
        save_price = []
        for row in spamreader:
            if(row[0] == etf_number ):
                save_price.append(float(row[6]))
                data_dateArray = datetime.datetime.strptime(row[1],"%Y%m%d")
                data_weekday = datetime.datetime.strftime(data_dateArray,'%w')
                data_weekly = datetime.datetime.strftime(data_dateArray,'%W')
                data_yearly = datetime.datetime.strftime(data_dateArray,'%Y')
                if ((data_yearly in dict_save_x_feature_yearly) == False):
                    dict_save_x_feature_weekly = {}
                    dict_save_y_feature_weekly = {}
                    dict_save_x_feature_yearly[data_yearly] = dict(dict_save_x_feature_weekly)
                    dict_save_y_feature_yearly[data_yearly] = dict(dict_save_y_feature_weekly)
                if ((data_weekly in dict_save_x_feature_weekly) == False):
                    feature_x_tmp = np.zeros(11)
                    feature_y_tmp = np.zeros(5)+-1
                    dict_save_x_feature_weekly[data_weekly] = list(feature_x_tmp)
                    dict_save_y_feature_weekly[data_weekly] = list(feature_y_tmp)
                if(int(data_weekday) != 0) and (int(data_weekday) != 6):
                    feature_x_tmp[int(data_weekday)-1] = float(row[6])
                    if(last_stock_price != 0) and (float(row[6]) >= last_stock_price ):
                        print(int(data_weekday))
                        feature_y_tmp[int(data_weekday)-1] = 1
                    else:
                        feature_y_tmp[int(data_weekday)-1] = 0
                    last_stock_price =  float(row[6])
                feature_x_tmp[5] = calculation_averge(save_price,5)
                feature_x_tmp[6] = calculation_averge(save_price,15)
                feature_x_tmp[7] = calculation_averge(save_price,30)
                feature_x_tmp[8] = calculation_averge(save_price,60)
                feature_x_tmp[9] = calculation_averge(save_price,120)
                feature_x_tmp[10] = calculation_averge(save_price,240)
                if(data_weekly in dict_save_x_feature_weekly):
                    dict_save_x_feature_weekly[data_weekly] = list(feature_x_tmp)
                    dict_save_y_feature_weekly[data_weekly] = list(feature_y_tmp)
                if(data_yearly in dict_save_x_feature_yearly):
                    dict_save_x_feature_yearly[data_yearly] = dict(dict_save_x_feature_weekly)
                    dict_save_y_feature_yearly[data_yearly] = dict(dict_save_y_feature_weekly)  
                data_index += 1
    return  dict_save_x_feature_yearly,dict_save_y_feature_yearly

def RNN(X,weights,biases):
    global n_inputs,n_steps,n_hidden_units,batch_size
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    lstm_cell =tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(cell=lstm_cell,inputs = X_in,dtype=tf.float32)
    outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    result = tf.matmul(outputs[-1],weights['out'])+biases['out']
    return result
def next_batch(step,x,y,batch_size):
    start = 0
    end = batch_size
    item_index = 0
    if(step+1 * batch_size < len(x)):   
        start = (step)*batch_size
        end = (step+1)*batch_size
    else:
        item_index = (step+1)%(int(len(x)/batch_size))
        start = (item_index)*batch_size
        end = (item_index+1)*batch_size
    return x[start:end],y[start:end]    
def LSTM_model(x_train_set_list,y_train_set_list):
    lr= 0.001
    training_steps = 500
    batch_size = 6
    n_inputs = 11
    n_steps = 1
    n_hidden_units = 1
    n_classes = 5
    x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
    y = tf.placeholder(tf.float32,[None,n_classes]) 
    weights = {
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
    }
    biases = {
    'in':tf.Variable(tf.constant(0.1,shape = [n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape =[n_classes,]))
    }
    pred = RNN(x,weights,biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    #correct_pred = tf.losses.mean_squared_error(y,pred)
    correct_pred = tf.equal(tf.round(pred),y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    init =  tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        batch_x,batch_y = next_batch(step,x_train_set_list,y_train_set_list,batch_size)
        train_x = np.reshape(x_train_set_list,(len(batch_x),n_steps,n_inputs))
        train_y = np.reshape(y_train_set_list,(len(batch_y),n_classes))
        while step < training_steps:
            sess.run([train_op],feed_dict = {x:train_x,y:train_y})
            print(sess.run(accuracy,feed_dict = {x:train_x,y:train_y}))
            step+=1
    
csv_name = 'Raw_Data/taetfp.csv'
load_csv(csv_name,'0050')
dict_save_x_feature_yearly,dict_save_y_feature_yearly = data_preprocesing(csv_name,'0050')    

x_train_set_list = []
y_train_set_list = []  

for key_index_yearly in dict_save_x_feature_yearly.keys():
    for key_index_weekly in dict_save_x_feature_yearly[key_index_yearly].keys():
        if(key_index_weekly+1 < len(dict_save_x_feature_yearly[key_index_yearly])) and (key_index_weekly+1) in dict_save_x_feature_yearly[key_index_yearly].keys() :
            if(min(dict_save_x_feature_yearly[key_index_yearly][key_index_weekly]) > 0) and (min(dict_save_y_feature_yearly[key_index_yearly][key_index_weekly+1]) > -1)
                x_train_set_list.append(dict_save_x_feature_yearly[key_index_yearly][key_index_weekly])
                y_train_set_list.append(dict_save_y_feature_yearly[key_index_yearly][key_index_weekly])

lr= 0.001
training_steps = 500
batch_size = 6
n_inputs = 11
n_steps = 1
n_hidden_units = 1
n_classes = 5                   
LSTM_model(x_train_set_list,y_train_set_list)
                
 