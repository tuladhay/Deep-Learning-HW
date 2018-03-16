HW2:

Given N training examples in 2 categories \{(\mathbf{x}_1, y_1),\ldots,(\mathbf{x}_N,y_N)\} { ( x 1 , y 1 ) , … , ( x N , y N ) } , your code should implement backpropagation using the cross-entropy loss (see Assignment 1 for the formula) on top of a sigmoid layer: (e.g. p(c_1|x)=\frac{1}{1+\exp(-f(x))}, p(c_2|x)=\frac{1}{1+\exp(f(x))} p ( c 1 | x ) = 1 1 + exp ⁡ ( − f ( x ) ) , p ( c 2 | x ) = 1 1 + exp ⁡ ( f ( x ) ) ), where you should train for an output f(x)=\mathbf{w}_{2}^\top g(\mathbf{W}_1^\top \mathbf{x}+b)+c f ( x ) = w 2 ⊤ g ( W 1 ⊤ x + b ) + c . LaTeX: g\left(x\right)=\max\left(x,0\right) g ( x ) = max ( x , 0 )  is the ReLU activation function (note Assignment #1 used a sigmoid activation but here it's ReLU), \mathbf{W}_1 W 1  is a matrix with the number of rows equal to the number of hidden units, and the number of columns equal to the input dimensionality.



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

