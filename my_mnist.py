import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_node = 784
hidden_node = 500
output_node = 10
alpha = 0.8 #0.8  #learning rate
beta = 0.999   #learning rate decay velocity
regularization_rate = 0.0001
batch_size = 1000
epoch_num = 1000 #1024

train_size = 55000
validation_size = 5000
# x: [60k, 28, 28],
# y: [60k]
(x, y),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

# shuffle before divide 1—fold validation
lable = np.arange(x.shape[0])
permutation = np.random.permutation(lable.shape[0])
x = x[permutation,:,:]
y = y[permutation]

# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
x_train = x[0:train_size,:,:]
y_train = y[0:train_size]
#1—fold validation
x_validation = x[train_size:60001,:,:]
y_validation = y[train_size:60001]
y_train_onehot = tf.one_hot(y_train, depth=output_node)
y_validation_onehot = tf.one_hot(y_validation, depth=output_node)
y_test_onehot = tf.one_hot(y_test, depth=output_node)

# data list for plot
loss_list = np.zeros([epoch_num,2],dtype = np.float64)
accuracy = np.zeros([epoch_num ,2],dtype= np.float64)
alpha_list = np.zeros([epoch_num],dtype= np.float64)

# return accuracy and predixtion of model
def prediction(x,y_onehot,w1,b1,w2,b2):
    x = tf.reshape(x, [-1, 28*28])
    z1 = x@w1 -tf.broadcast_to(b1, [x.shape[0], w1.shape[1]])
    h1 = tf.nn.relu(z1)
    # h1 = tf.nn.sigmoid(z1)
    z2 = h1@w2 - tf.broadcast_to(b2, [h1.shape[0], w2.shape[1]])
    out = tf.nn.sigmoid(z2)
    pre = tf.argmax(out,1)
    correct_prediction = tf.equal(pre,tf.argmax(y_onehot,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100

    # compute loss
    # mse = mean(sum(y-out)^2)  mean: scalar  => RMSE
    # loss_mse = tf.square(y_onehot - out)
    # loss_rmse = tf.reduce_mean(loss_mse)

    #sfot max => cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= out , labels= tf.argmax(y_onehot,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #L2 ||w||
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    regularization_mean = regularization_rate * tf.reduce_mean(regularization)

    # loss = loss_rmse + regularization_mean
    loss = cross_entropy_mean + regularization_mean

    return pre,accuracy,loss;

train_db = tf.data.Dataset.from_tensor_slices(\
    (x_train,y_train_onehot)).batch(batch_size)
train_db.shuffle(55000)
train_iter = iter(train_db)
sample = next(train_iter)


# random initialize weights and bias
w1 = tf.Variable(tf.random.truncated_normal([input_node,\
                                             hidden_node], stddev=0.1))  # stddev = 0.1
b1 = tf.Variable(tf.zeros([hidden_node]))
w2 = tf.Variable(tf.random.truncated_normal([hidden_node,\
                                             output_node], stddev=0.1))
b2 = tf.Variable(tf.zeros([output_node]))

# train main
for epoch in range(epoch_num): # iterate db for 10
    for step, (x_train, y_train_onehot) in enumerate(train_db): # for every batch
        # x:[128, 28, 28]
        # y: [128]

        # [b, 28, 28] => [b, 28*28]
        x_train = tf.reshape(x_train, [-1, 28*28])

        with tf.GradientTape() as tape: # tf.Variable
            z1 = x_train@w1 - tf.broadcast_to(b1, [x_train.shape[0], hidden_node])
            # h1 = tf.nn.sigmoid(z1)
            h1 = tf.nn.relu(z1)
            z2 = h1@w2 - tf.broadcast_to(b2, [h1.shape[0], output_node])
            out = tf.nn.sigmoid(z2)

            # compute loss
            # mse = mean(sum(y-out)^2)  mean: scalar  => RMSE
            # loss_mse = tf.square(y_train_onehot - out)
            # loss_rmse = tf.reduce_mean(loss_mse)

            #sfot max => cross entropy
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= out , labels= tf.argmax(y_train_onehot,1))
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

            #L2 ||w||
            regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
            regularization_mean = regularization_rate * tf.reduce_mean(regularization)

            # loss = loss_rmse + regularization_mean
            loss = cross_entropy_mean + regularization_mean

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2])
        #learning rate damping
        alpha_epoch  =  alpha * math.pow(beta,epoch)
        alpha_list[epoch] = alpha_epoch
        # update w and b
        w1.assign_sub(alpha_epoch * grads[0])
        b1.assign_sub(alpha_epoch * grads[1])
        w2.assign_sub(alpha_epoch * grads[2])
        b2.assign_sub(alpha_epoch * grads[3])


    # every epoch print acc & loss
    (_,accuracy[epoch,0],loss_list[epoch,0]) = prediction(x_train,y_train_onehot,w1,b1,w2,b2)
    (_,accuracy[epoch,1],loss_list[epoch,1])= prediction(x_validation,y_validation_onehot,w1,b1,w2,b2)
    print('epoch{}'.format(epoch),':TrainAccuracy:%.8f%%'%(accuracy[epoch,0]), \
          'ValAccuracy:%.8f%%'%(accuracy[epoch,1],),'TrainLoss:{}'.format(loss_list[epoch,0]), \
          'ValLoss:{}'.format(loss_list[epoch,1]))



#plot epoch  accuracy curve
fig= plt.figure(1)
t = np.linspace(1,epoch_num,epoch_num,endpoint = True)
plt.plot(t, accuracy[...,0], '-',color = 'red',label='Train Accuracy')
plt.plot(t, accuracy[...,1], '--',color = 'blue',label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Epoch Accuracy Curve',verticalalignment='bottom')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
fig.savefig('epoch_accuracy.png',verticalalignment='bottom')

plt.show()

#plot epoch  loss curve
fig= plt.figure(2)
l1,=plt.plot(t, loss_list[...,0], '-',color = 'red',label='Train Loss')
l2,=plt.plot(t, loss_list[...,1], '--',color = 'blue',label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Epoch Loss Curve',verticalalignment='bottom')
plt.xlabel('Epoch')
plt.ylabel('Loss')
fig.savefig('epoch_loss.png',verticalalignment='bottom')
plt.show()

#plot curve of learning rate
fig = plt.figure(4)
plt.plot(t, alpha_list, '-',color = 'green',label='Alpha')
plt.title('Declay Alpha Curve',verticalalignment='bottom')
fig.savefig('alpha.png')
plt.show()

#plot test images
actuals = y_test[0:80]
(predictions,acc,_)= prediction(x_test[0:80,:,:],y_test_onehot[0:80,:],w1,b1,w2,b2)
#predictions = predictionss[0:100]
images = np.squeeze(x_test[0:80,:,:])
Nrows = 8
Ncols = 10
for i in range(80):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(images[i], [28,28]), cmap='Greys_r')
    #plt.title('Actual: ' + str(int(actuals[i])) + ' Pred: ' + str(int(predictions[i])),
    #          fontsize=10)

    plt.title('Label: ' + str(int(actuals[i])),\
              fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
fig.savefig('test image.png')
plt.show()
