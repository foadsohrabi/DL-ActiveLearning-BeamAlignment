import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
from func_codedesign_cont import func_codedesign_cont
import scipy.io as sio
from keras.layers import Dense, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
'System Information'
N = 64   #Number of BS's antennas
delta_inv = 128 #Number of posterior intervals inputed to DNN 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
tau = int(2*S) #Pilot length
OS_rate = 20 #Over sampling rate in each AoA interval (N_s in the paper)
delta_inv_OS = OS_rate*delta_inv #Total number of AoAs for posterior computation
delta_OS = 1/delta_inv_OS 
'Channel Information'
phi_min = -60*(np.pi/180) #Lower-bound of AoAs
phi_max = 60*(np.pi/180) #Upper-bound of AoAs
num_SNR = 8 #Number of considered SNRs
low_SNR_idx = 4 #Index of Lowest SNR for training
high_SNR_idx = 8 ##Index of highest SNR for training + 1 
idx_SNR_val = 7 #Index of SNR for validation (saving parameters)
snrdBvec = np.linspace(start=-10,stop=25,num=num_SNR) #Set of SNRs
Pvec = 10**(snrdBvec/10) #Set of considered TX powers
mean_true_alpha = 0.0 + 0.0j #Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5) #STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5) #STD of the Gaussian noise per real dim.
#####################################################
'Learning Parameters'
initial_run = 1 #0: Continue training; 1: Starts from the scratch
n_epochs = 10000 #Num of epochs
learning_rate = 0.0001 #Learning rate
batch_per_epoch = 10 #Number of mini batches per epoch
batch_size_order = 2 #Mini_batch_size = batch_size_order*delta_inv
val_size_order = 25 #Validation_set_size = val_size_order*delta_inv
scale_factor = 17*2 #Scaling the number of tests
test_size_order = 23 #Test_set_size = test_size_order*delta_inv*scale_factor
######################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Define initialization method
######################################## Place Holders
alpha_input = tf.placeholder(tf.complex64, shape=(None,1), name="alpha_input")
phi_input = tf.placeholder(tf.float32, shape=(None,), name="phi_input")
######################################################
#Constructing the array responses for AoA samples
A_BS, phi_set, A_BS_OS, phi_OS_set  =  func_codedesign_cont(delta_inv,delta_inv_OS,phi_min,phi_max,N)
##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    ###############
    from0toN = tf.cast(tf.range(0, N, 1),tf.float32)
    #### Actual Channel
    phi = tf.reshape(phi_input,[-1,1])
    h_act = {0: 0}
    hR_act = {0: 0}
    hI_act = {0: 0}   
    phi_expanded = tf.tile(phi,(1,N))
    a_phi = (tf.exp(1j*np.pi*tf.cast(tf.multiply(tf.sin(phi_expanded),from0toN),tf.complex64)))
    
with tf.name_scope("channel_sensing"):
    posteriors_OS = delta_OS*tf.ones(shape=[tf.shape(phi_input)[0],delta_inv_OS],dtype=tf.float32)
    posteriors = tf.reduce_sum(tf.reshape(posteriors_OS,[-1,delta_inv,OS_rate]),axis=2)
       
    A1 = tf.get_variable("A1",  shape=[delta_inv+2,1024], dtype=tf.float32, initializer= he_init)
    A2 = tf.get_variable("A2",  shape=[1024,1024], dtype=tf.float32, initializer= he_init)
    A3 = tf.get_variable("A3",  shape=[1024,1024], dtype=tf.float32, initializer= he_init)
    A4 = tf.get_variable("A4",  shape=[1024,2*N], dtype=tf.float32, initializer= he_init)
    b1 = tf.get_variable("b1",  shape=[1024], dtype=tf.float32, initializer= he_init)
    b2 = tf.get_variable("b2",  shape=[1024], dtype=tf.float32, initializer= he_init)
    b3 = tf.get_variable("b3",  shape=[1024], dtype=tf.float32, initializer= he_init)
    b4 = tf.get_variable("b4",  shape=[2*N], dtype=tf.float32, initializer= he_init)

    w_dict = []
    y_dict = [] 
    for t in range(tau):
        snr = lay['P']*tf.ones(shape=[tf.shape(phi_input)[0],1],dtype=tf.float32)
        snr_dB = tf.log(snr)/np.log(10)
        snr_normal = (snr_dB-1)/np.sqrt(1.6666) #Normalizing for the range -10dB to 30dB
        
        iter_idx = t*tf.ones(shape=[tf.shape(phi_input)[0],1],dtype=tf.float32)
        iter_idx_normal = (iter_idx -6.5)/np.sqrt(16.25) #Normalizing for the range 0 to 13
        
        posteriors_normal = (posteriors-0.5)/np.sqrt(1/12) #Normalizing for the range 0 to 1
        
        'DNN designs the next sensing direction'
        dnn_input = tf.concat([posteriors_normal,snr_normal,iter_idx_normal],axis=1)        
        x1 = tf.nn.relu(dnn_input@A1+b1)
        x1 = BatchNormalization()(x1)
        x2 = tf.nn.relu(x1@A2+b2)
        x2 = BatchNormalization()(x2)
        x3 = tf.nn.relu(x2@A3+b3)
        x3 = BatchNormalization()(x3)
        w_her_unnorm = x3@A4+b4        
        w_her_r = w_her_unnorm[:,0:N]
        w_her_i = w_her_unnorm[:,N:2*N]                
        abs_w_her_ri = tf.sqrt(tf.pow(w_her_r, 2) + tf.pow(w_her_i, 2))
        w_her_r = (1/np.sqrt(N))*tf.divide(w_her_r,abs_w_her_ri)
        w_her_i = (1/np.sqrt(N))*tf.divide(w_her_i,abs_w_her_ri) 
        
        w_her = tf.concat([w_her_r,w_her_i],axis=1)
        
        w_norm = tf.reshape(tf.norm(w_her,axis=1),(-1,1))
        #w_her = tf.divide(w_her,w_norm)
        w_her_complex = tf.complex(w_her[:,0:N],w_her[:,N:2*N])
        w_dict.append(w_her_complex)
        W_her = tf.stack(w_dict,axis=1)
        'BS observes the next measurement'
        y_noiseless = tf.reduce_sum( tf.multiply( w_her_complex, a_phi), 1, keepdims=True )
        noise =  tf.complex(tf.random_normal(tf.shape(y_noiseless), mean = 0.0, stddev = noiseSTD_per_dim),\
                            tf.random_normal(tf.shape(y_noiseless), mean = 0.0, stddev = noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']),0.0)*tf.multiply(y_noiseless,alpha_input) + noise                
        y_dict.append(y_complex)
        Y = tf.reshape(tf.stack(y_dict,axis=1),[-1,t+1])
        Y_tile = tf.tile(tf.reshape(Y,[-1,t+1,1]),(1,1,delta_inv_OS))  
        
        'BS estimates alpha by MMSE'
        c = tf.complex(tf.sqrt(lay['P']),0.0)*\
            tf.reshape(tf.reshape(W_her, [-1, N]) @ A_BS_OS, [-1, t+1, delta_inv_OS])
        cT = tf.transpose(tf.conj(c),[0,2,1])
        coef = tf.cast(tf.reduce_sum(tf.pow(tf.abs(c),2),axis=1),tf.complex64)

        cT_Y_inner_prod = (tf.matmul(cT,Y[..., None]))[..., 0]
        
        alpha_hat = tf.divide(cT_Y_inner_prod,coef+2*noiseSTD_per_dim**2)
        alpha_hat_extended = tf.tile(tf.reshape(alpha_hat,[-1,1,delta_inv_OS]),(1,t+1,1))
        
        'BS updates the posterior distribution'
        mean_complex = tf.multiply(alpha_hat_extended,c)
        unnorm_post = tf.exp(-tf.pow(tf.abs(Y_tile - mean_complex),2))
        posteriors_temp = tf.math.reduce_prod(unnorm_post,axis=1)
        sum_posteriors_temp = tf.reshape(tf.reduce_sum(posteriors_temp,axis=1),[-1,1])
        posteriors_OS = tf.divide(posteriors_temp,sum_posteriors_temp+0.00000000001)
        posteriors = tf.reduce_sum(tf.reshape(posteriors_OS,[-1,delta_inv,OS_rate]),axis=2)
    
    phi_hat = Dense(units=1, activation='linear')(posteriors_OS)   
#####################################################################################
####### Loss Function  
loss = tf.keras.losses.MSE(phi_input ,phi_hat[:,0])
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
##########################################################################
###########  Validation Set
alpha_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[val_size_order*delta_inv,1])\
            +1j*np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[val_size_order*delta_inv,1])            
phi_val = np.random.uniform(low=phi_min, high=phi_max, size=delta_inv*val_size_order)
feed_dict_val = {alpha_input: alpha_val,
                  phi_input: phi_val,
                  lay['P']: Pvec[idx_SNR_val]}
#############  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './params')
    best_loss = sess.run(loss, feed_dict=feed_dict_val)
    print(best_loss)
    print(tf.test.is_gpu_available())
    for epoch in range(n_epochs):
        if epoch == 0:
            best_loss = sess.run(loss, feed_dict=feed_dict_val)           
            print(best_loss)
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            idx_temp = np.random.randint(low=low_SNR_idx, high=high_SNR_idx, size=1)
            snr_temp = snrdBvec[idx_temp[0]]
            #snr_temp = np.random.uniform(low=20, high=30, size=[1])
            P_temp = 10**(snr_temp/10)
            alpha_batch = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_order*delta_inv,1])\
                        +1j*np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_order*delta_inv,1])          
            phi_batch = np.random.uniform(low=phi_min, high=phi_max, size=delta_inv*batch_size_order)
            feed_dict_batch = {alpha_input: alpha_batch,
                              phi_input: phi_batch,
                              lay['P']: P_temp}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1

        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch',epoch,'  loss_test:%2.5f'%loss_val,'  best_test:%2.5f'%best_loss)    
        if epoch%10==9: #Every 10 iterations it checks if the validation performace is improved, then saves parameters
            if loss_val < best_loss:
                save_path = saver.save(sess, './params')
                best_loss = loss_val

###########  Final Test    
    performance = np.zeros([len(snrdBvec),scale_factor])
    for j in range(scale_factor):
        print(j)
        alpha_test = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[test_size_order*delta_inv,1])\
                    +1j*np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[test_size_order*delta_inv,1]) 
        phi_test = np.random.uniform(low=phi_min, high=phi_max, size=delta_inv*test_size_order)
        for i in range(len(snrdBvec)):
            feed_dict_test = {alpha_input: alpha_test,
                                    phi_input: phi_test,
                                    lay['P']: Pvec[i]}
            mse_loss,phi_hat_test= sess.run([loss,phi_hat],feed_dict=feed_dict_test)
            performance[i,j] = mse_loss
            
    performance = np.mean(performance,axis=1)       
            
######### Plot the test result 
plt.semilogy(snrdBvec, performance)        
plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel('Average MSE')

sio.savemat('data_DNN_unknown_alpha_MMSE_updatePIs_EST_hyb_norm.mat',dict(performance= performance,\
                                       snrdBvec=snrdBvec,N=N,delta_inv=delta_inv,\
                                       mean_true_alpha=mean_true_alpha,\
                                       std_per_dim_alpha=std_per_dim_alpha,\
                                       noiseSTD_per_dim=noiseSTD_per_dim, tau=tau))

