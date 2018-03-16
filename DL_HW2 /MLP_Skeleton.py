"""
Yathartha Tuladhar
"""

from __future__ import division
from __future__ import print_function

import sys

import cPickle
import numpy as np
import matplotlib.pyplot as plt


# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def forward(self, x, weights, bias):
        z = x.dot(weights) + bias
        return z

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        pass


class ReLU(object):
    def forward(self, x):
        return x * (x > 0)

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        pass
    # DEFINE backward function
    # ADD other operations in ReLU if needed


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
# Implement backpropagation using the cross-entropy loss (see Assignment 1 for the formula) on top of a sigmoid layer
class SigmoidCrossEntropy(object):

    def forward(self, x, labels):
        # x = x/max(x)
        prediction = 1.0/(1.0 + np.exp(-x))
        # print(x,prediction)

        prediction = prediction + (prediction==0)*(np.zeros((prediction.shape[0], prediction.shape[1])) + 0.001)
        prediction = prediction + (prediction == 1)*(np.zeros((prediction.shape[0], prediction.shape[1])) - 0.001)

        loss = -1.0*( labels*np.log(prediction) + (1.0-labels)*np.log(1.0-prediction) )
        # avg_loss = sum(loss)/labels.shape[0]
        return loss, prediction

    def backward(
        self,
        grad_output,# dz
        h_out,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
        ):
        dW = h_out.T.dot(grad_output) # dw = h.dz
        # dW = grad_output.dot(h_out)
        return dW

    # DEFINE backward function
    # ADD other operations and data entries in SigmoidCrossEntropy if needed


def extrude(w, batch_size):
    ig = np.copy(w)
    for i in range(batch_size-1):
        w = np.concatenate((w, ig), axis=0)

    return w



class MLP(object):

    def __init__(self, i_dims, h_units):
        i_dim = i_dims #3072
        h_units = h_units
        self.eps = 0.001
        self.W1 = np.random.uniform(low=0-self.eps, high=0+self.eps, size=(i_dims, h_units) )
        self.W2 = np.random.uniform(low=0- self.eps, high=0+self.eps, size=(h_units, 1) )
        self.b1 = np.random.uniform(low=0- self.eps, high=0+self.eps, size=(1, h_units) )
        self.b2 = np.random.uniform(low=0- self.eps, high=0+self.eps, size=(1, 1) )

        #Mom
        self.mom_W1 = np.random.uniform(low=0-self.eps, high=0+self.eps, size=(i_dims, h_units) ) * 0
        self.mom_W2 = np.random.uniform(low=0- self.eps, high=0+self.eps, size=(h_units, 1) ) *0
        self.mom_b1 = np.random.uniform(low=0- self.eps, high=0+self.eps, size=(1, h_units) ) *0
        self.mom_b2 = np.random.uniform(low=0- self.eps, high=0+self.eps, size=(1, 1) ) *0

        # self.b1 *=0
        # self.b2 *=0

        print(self.W1.shape)
        print(self.W2.shape)
        print(self.b1.shape)
        print(self.b2.shape)
        # self.linear_transform1 = LinearTransform(self.W1,self.b1)
        # self.linear_transform2 = LinearTransform(self.W2,self.b2)


    def train(
        self,
        x_batch,
        y_batch,
        learning_rate,
        momentum,
        l2_penalty,
        ):

        b1 = extrude(np.copy(self.b1), batch_size=16)
        b2 = extrude(np.copy(self.b2), batch_size=16)

        # Linear transform 1
        # z1 = LinearTransform.forward(x_batch)
        z1 = x_batch.dot(self.W1) + b1

        # ReLU
        # activation1 = ReLU()
        z1 = z1 * (z1 > 0)
        hidden_output = np.copy(z1)

        # Linear Transform 2
        z2 = hidden_output.dot(self.W2) + b2
        # activation2 = SigmoidCrossEntropy()

        # Sigmoid Cross Entropy
        sigmd = SigmoidCrossEntropy()
        loss, predict = sigmd.forward(z2, y_batch)      # Loss is an array
        # predict is the sigmoid(z2), or 'a' in some literature, or y^ or t
        avg_loss = sum(loss)

        # Forward Pass Done
        # Now do Backward Pass


        dz = predict - y_batch  # this should be grad output (a - y)
        d_W2 = sigmd.backward(dz, hidden_output, learning_rate, momentum, l2_penalty)
        d_b2 = dz

        # Output layer backward done

        dhidden = np.dot(dz, self.W2.T) # Relu backprop
        dhidden[hidden_output <= 0] = 0

        d_W1 = np.dot(x_batch.T, dhidden)
        d_b1 = dhidden



        self.mom_W2 = momentum * self.mom_W2 - learning_rate * d_W2
        self.mom_b2 = momentum * self.mom_b2 - learning_rate * np.reshape(np.mean(d_b2, axis=0), (1, len(d_b2[0])))
        self.mom_W1 =  momentum * self.mom_W1 - learning_rate * d_W1
        self.mom_b1 = momentum * self.mom_b1 - learning_rate * np.reshape(np.mean(d_b1, axis=0), (1, len(d_b1[0])))


        self.W2 += self.mom_W2 - 0.001*self.W2
        self.b2 += self.mom_b2 - 0.001*self.b2
        self.W1 += self.mom_W1 - 0.001*self.W1
        self.b1 += self.mom_b1 - 0.001*self.b1

        # self.W2 = self.W2 - learning_rate * d_W2
        # self.b2 = self.b2 - learning_rate * np.reshape(np.mean(d_b2, axis=0), (1, len(d_b2[0])))
        # self.W1 = self.W1 - learning_rate * d_W1
        # self.b1 = self.b1 - learning_rate*  np.reshape(np.mean(d_b1, axis=0), (1, len(d_b1[0])))



        # print("d_w2 " + str(self.W2))
        # print("d_w1 " + str(self.W1))

        return avg_loss



    def evaluate(self, x, y):

        b1 = extrude(np.copy(self.b1), batch_size=len(x))
        b2 = extrude(np.copy(self.b2), batch_size=len(x))

        # Linear transform 1
        z1 = x.dot(self.W1) + b1



        # ReLU
        z1 = z1 * (z1 > 0)
        hidden_output = np.copy(z1)

        # Linear Transform 2
        z2 = hidden_output.dot(self.W2) + b2
        # activation2 = SigmoidCrossEntropy()

        # Sigmoid Cross Entropy
        sigmd = SigmoidCrossEntropy()
        loss, predict = sigmd.forward(z2, y)  # Loss is an array
        # predict is the sigmoid(z2), or 'a' in some literature, or y^ or t
        avg_loss = sum(loss)

        diff = predict - y  # if prediction is same as labels diff will be zero
        is_correct = (np.abs(diff))<=0.49

        accuracy = np.mean(is_correct) * 100.0


        #predict = (predict>=0.5)*np.ones((predict.shape[0], predict.shape[1]))
        #predict = (predict<0.5)*np.zeros((predict.shape[0], predict.shape[1]))


        # incorrect = np.count_nonzero(diff)
        # accuracy = 1.0 - (incorrect/diff.shape[0])
        # accuracy*=100

        return accuracy, avg_loss



        # Forward Pass Done


# To generate mini-batches for training
# https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':

    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']


    # train_x = train_x[0:10]
    # train_y = train_y[0:10]
    num_examples, input_dims = train_x.shape
    # 10000, 3072

    num_epochs = 100
    num_batches = 16
    hidden_units = 10
    learning_rate = 0.001
    momentum = 0.6
    l2_penalty = 1          # CHOOSE A BETTER GUESS
    mlp = MLP(input_dims, hidden_units)         # This is our Neural Network

    loss_per_epoch = []
    train_accuracy_per_epoch = []
    test_accuracy_per_epoch = []

    for epoch in xrange(num_epochs):
            # EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss
            total_loss = 0.0
            # print("New Epoch Start ")
            for batch in iterate_minibatches(train_x, train_y, num_batches, shuffle=True):
                # if num_batches is 10, this will run 10000/10 = 1000 times
                # total_loss = 0.0
                avg_loss = 0.0
                x_batch, y_batch = batch
                x_batch = x_batch/255.0

                avg_loss = mlp.train(x_batch, y_batch, learning_rate, momentum, l2_penalty)

                total_loss = total_loss + avg_loss

                # print("Total_loss = " + str(total_loss))
            print("Epoch total_loss " + str(total_loss))

            loss_per_epoch.append(total_loss)

            # Now, calculate train_accuracy
            train_accuracy, train_loss = mlp.evaluate(train_x, train_y)
            train_accuracy_per_epoch.append(train_accuracy)
            print(train_accuracy, train_loss)

            test_accuracy, test_loss = mlp.evaluate(test_x, test_y)
            test_accuracy_per_epoch.append(test_accuracy)
            print(test_accuracy, test_loss)

    # plotting after all epochs are done
    plt.plot(loss_per_epoch)
    plt.title('Average Loss (' + "Ep:" + str(num_epochs) + " Batches: "+str(num_batches)+" H-units:"+str(hidden_units)+" LR:"+str(learning_rate) + " M:" + str(momentum) + " )")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()

    plt.plot(train_accuracy_per_epoch)
    plt.title('Training Accuracy (' + "Ep:" + str(num_epochs) + " Batches: "+str(num_batches)+" H-units:"+str(hidden_units)+" LR:"+str(learning_rate) + " M:" + str(momentum) + " )")
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.show()

    plt.plot(test_accuracy_per_epoch)
    plt.title('Test Accuracy (' + "Ep:" + str(num_epochs) + " Batches: " + str(num_batches) + " H-units:" + str(
        hidden_units) + " LR:" + str(learning_rate) + " M:" + str(momentum) + " )")
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy')
    plt.show()

    print("Finished Plotting")

        #
        #     print(
        #         '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
        #             epoch + 1,
        #             b + 1,
        #             total_loss,
        #         ),
        #         end='',
        #     )
        #     sys.stdout.flush()
        #
        # # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        # print()
        # print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
        #     train_loss,
        #     100. * train_accuracy,
        # ))
        # print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
        #     test_loss,
        #     100. * test_accuracy,
        # ))
