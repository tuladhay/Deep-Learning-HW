import cPickle as pickle
import matplotlib.pyplot as plt

class SaveVal():
    def __init__(self):
        self.testLoss = []
        self.trainLoss = []
        self.trainAcc = []
        self.testAcc = []

# **********************   Part 1
# **********************   Batch Normalization
# Without the class, pickle will not load the data
# saved_data = pickle.load(open("part1_save_test_train_loss_acc_BN.p", "rb"))
# print
# epochs = [e+1 for e in range(len(saved_data.testAcc))]
# print(epochs)
#
# # Test Accuracy and Train accuracy
# plt.plot(epochs, saved_data.testAcc, label="Test")
# plt.plot(epochs, saved_data.trainAcc, label="Train")
#
# plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=2.)
# plt.axis([1, max(epochs), 0, 1])
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Train/Test accuracy vs epochs\nPart 1 Batch Normalization")
# plt.show()
#
# # Test and Train loss
# plt.plot(epochs, saved_data.testLoss, label="Test")
# plt.plot(epochs, saved_data.trainLoss, label="Train")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=2.)
# plt.axis([1, max(epochs), 0, 1])
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Train/Test loss vs epochs\nPart 1 Batch Normalization")
# plt.show()


# *************************** Without BN
filename = "part4_save_test_train_loss_acc_BNtwice_addedFC_AdamLR0005_ep25.p"
saved_data = pickle.load(open(filename, "rb"))
print
epochs = [e+1 for e in range(len(saved_data.testAcc))]
print(epochs)

# Test Accuracy and Train accuracy
plt.plot(epochs, saved_data.testAcc, label="Test")
plt.plot(epochs, saved_data.trainAcc, label="Train")

plt.legend(bbox_to_anchor=(1.05, 0.5), loc=1, borderaxespad=2.)
plt.axis([1, max(epochs), 0, 1.1])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train/Test accuracy vs epochs\nPart 4 Batch Norm*2, FC 512 + Conv, Adam Optimizer")
plt.show()

# Test and Train loss
plt.plot(epochs, saved_data.testLoss, label="Test")
plt.plot(epochs, saved_data.trainLoss, label="Train")
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=2.)
plt.axis([1, max(epochs), 0, 2])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train/Test loss vs epochs\nPart 4 Batch Norm*2, FC 512 + Conv, Adam Optimizer")
plt.show()
