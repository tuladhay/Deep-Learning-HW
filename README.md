HW2:

Given N training examples in 2 categories, your code should implement backpropagation using the cross-entropy loss on top of a sigmoid layer. The hidden layer should have a ReLU activation function.

Write a function that evaluates the trained network, as well as computes all the subgradients.

Write a function that performs stochastic mini-batch gradient descent training.

Train the network on the attached 2-class dataset extracted from CIFAR-10: (data can be found in the cifar-2class-py2.zip file on Canvas.). The data has 10,000 training examples in 3072 dimensions and 2,000 testing examples. For this assignment, just treat each dimension as uncorrelated to each other. Train on all the training examples, tune your parameters (number of hidden units, learning rate, mini-batch size, momentum) until you reach a good performance on the testing set. What accuracy can you achieve? 

Training Monitoring: For each epoch in training, your function should evaluate the training objective, testing objective, training misclassification error rate (error is 1 for each example if misclassifies, 0 if correct), testing misclassification error rate (5 points).

Tuning Parameters: please create three figures with following requirements. Save them into jpg format:
i) test accuracy with different number of batch size
ii)test accuracy with different learning rate
iii) test accuracy with different number of hidden units

Discussion about the performance of your neural network.



HW3:

1) Add a batch normalization layer after the first fully-connected layer(fc1)
Save the model after training(Checkout our tutorial on how to save your model).
Becareful that batch normalization layer performs differently between training and evalation process, make sure you understand how to convert your model between training mode and evaluation mode(you can find hints in my code).
Observe the difference of final training/testing accuracy with/without batch normalization layer.

2) Modify our model by adding another fully connected layer with 512 nodes at the second-to-last layer (before the fc2 layer)
Apply the model weights you saved at step 1 to initialize to the new model(only up to fc2 layer since after that all layers are newly created) before training. Train and save the model (Hint: check the end of the assignment description to see how to partially restore weights from a pretrained weights file).

3) Try to use an adaptive schedule to tune the learning rate, you can choose from RMSprop, Adagrad and Adam (Hint: you don't need to implement any of these, look at Pytorch documentation please)

4) Try to tune your network in two other ways (10 points) (e.g. add/remove a layer, change the activation function, add/remove regularizer, change the number of hidden units, more batch normalization layers) not described in the previous four. You can start from random initialization or previous results as you wish.

For each of the settings 1) - 4), please submit a PDF report your training loss, training accuracy, validation loss and validation accuracy. Draw 2 figures for each of the settings 1) - 4) (2 figures for each different tuning in (4)) with the x-axis being the epoch number, and y-axis being the loss/accuracy, use 2 different lines in the same figure to represent training loss/validation loss, and training accuracy/validation accuracy.

