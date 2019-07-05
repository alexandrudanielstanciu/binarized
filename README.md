# binarized
VGG16 and MobilenetV2 with binary weights and activations

Credits to DingKe for binary ops and binarized layers.
https://github.com/DingKe/nn_playground/tree/master/binarynet

I have created custom implementation of VGG16 and MobileNetV2 as in BinaryNet.
New custom layer was added for SeparableConv2D.

### To run the scripts (Keras 2.1.2):
python binary_train.py
