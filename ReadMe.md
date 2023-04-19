<!--
###############################################################################################################################################################
＃　this file is to decribe the data pre-processing process and Model select in image_classification_nn.py
###############################################################################################################################################################
function in used:

function FeatureExtractorGrayscale is used to load the images and covert to grayscale. 

function top1_5_score is used to calculate the TOP-1, TOP-5 Accuracy from the prediction of each model.
###############################################################################################################################################################
model selection:

LeNet network obj is called from */model/network

the net.train() function is to train the network after calling the model obj.
by default, the input image sizes = 64*64*1 batch size = 256, epoch = 1

model/network.py

this file contain the customzied LeNet model.
learning rate is fixed and set as 1-e3.
kernel size is set as 3, can be modified by youself, but be aware of the input shape after changing kernel size or adding a new layer.
activation function using by default is Sigmoid*x, also known as Swish function. Other activation function such as Relu, Sigmoid were defined in the layer.py.
pickle package is imported to save the weight of model and the loss and accuracy to local after training. You can reload the model after training. 

model/layers.py

nn layers and the forward/ backward functions in the LeNet model are defined as obj in the file
take a look at the Sigmoid and Swish function. the float overflow issue has been fixed by restricted the decimals with the package. 
if there is any other way to fix this issue, opeing dicussion are welcome to every of you guys.

model/loss.py

loss function using in the customized LeNet model. Cross_entropy is selected in this case.
 --> 
