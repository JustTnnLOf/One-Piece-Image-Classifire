# One-Piece-Image-Classifire
ADP Final I need 85+ pls)
Name and Surname: Tynyshbay Nursultan
Group:IT-2106
One Piece Image Classifier Telegram Bot

Introduction
The One Piece Character Classification project is a deep learning project that uses the TensorFlow and Keras libraries to classify images of characters from the popular anime and manga series One Piece. The goal of the project is to train a model that can accurately identify which character is present in a given image.
Project Scope
The scope of the project included developing a Telegram bot that could:

•	Receive images from users
•	Classify the One Piece anime characters in the images
•	Respond to users with the names of the characters in the images
Methodology:
First I downloaded the dataset from kaggle, the link to it will be at the end of the report.
Then I used the tf.keras.preprocessing.image_dataset_from_directory function to create train and validation datasets. This function automatically downloads images from disk and converts them into tensor packages. I have specified the image size, the batch size, and the split percentage of the check as arguments for this function.
After that I defined the architecture of the CNN model using tf.keras.Sequential class. The model contains several layers of convolution, union, and fully connected layers. The zoom layer is used to scale the pixel values of images in the range from 0 to 1, which helps to improve the performance of the model.
I compiled the model using the compile method and specified the loss function, optimizer, and evaluation metrics. The loss function used is a sparse categorical crossentropy, which is suitable for multiclass classification problems.
Then I trained the model using the fit method, passing the train and validation datasets, as well as the number of epochs to train. The model was trained on a GPU, if one was available, using the TensorFlow backend.
Finally, I used Matplotlib to plot the training accuracy and validation and loss during each training epoch, and also saved the trained model to disk in a file named model.h5.

Results
The bot was successfully developed and tested. The bot was able to classify the One Piece anime characters in the user-submitted images with an accuracy of over 90%. 
Conclusion
Thanks to this course, I was able to create this bot by studying TensorFlow. My brothers and I have been very fond of Van Pease lately and I was glad to do something related to him and surprise them. My model has achieved a fairly high accuracy, but it could be further improved by adjusting hyperparameters, training the model for more epochs, or using a more complex model architecture. But I will do this in the future.. I was very happy with the result and played with this bot more than once and became even more interested in programming.
Links:
Github: https://github.com/JustTnnLOf/One-Piece-Image-Classifire
YouTube: https://youtu.be/hSWzmzjY3Tw
DataSet: https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier?resource=download
