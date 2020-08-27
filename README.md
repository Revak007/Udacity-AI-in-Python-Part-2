# Udacity-AI-in-Python-Part2
This repository contains part 2 of the final project for the AI in Python nanodegree. In this part, we create a command line application to train our image classifier 
and use it to predict the class of an image. Python 3.7 is used here. Most of the same packages are required for part 1 and part 2, including pytorch and numpy. One additional package required is pillow, which is further described in the installation section.

## INSTALLATION
-	First, clone this repository:
    `Git clone https://github.com/Revak007/Udacity-AI-in-Python-Part2.git`
    
Feel free to skip the below part if you have already installed all packages from part 1, and also the pillow package
    -	 Then to install Anaconda go to the following link, find your operating system, and follow the instructions: https://docs.anaconda.com/anaconda/install/
    -	 Then, go to the Anaconda prompt and install necessary packages:
            `Conda install numpy, pandas, pytorch, matplotlib, pillow`
            -NOTE: pillow is so that you can later use PIL to open a PIL image

## USAGE
NOTE: For any paths required, make sure the folder is in the current/working directory. Example, the flowers directory is in the ImageClassifier folder which is the current directory so we can just do “python train.py flowers”

1)	Training: 
The train_helper.py script consists of helper functions that are used to train the classifier in train.py. Thus, both train_helper and train.py are necessary parts. The command line takes multiple arguments, including dataset and hyperparameters:

- First, the required arguments are:
	 `python train.py data_directory_path`
- Option 1 is entering a checkpoint path (by what name to save checkpoint/features of model).    Default is ‘checkpoint.pth’:			
   `python train.py data_directory_path --save_dir ‘checkpoint.pth’`
- Option 2 is the type of architecture you want. The options are ‘vgg’, ‘resnet’, and ‘alexnet’. Default is ‘vgg’:
	 `python train.py data_directory_path --arch ‘vgg’`
- Option 3 is the model learning rate. The smaller learning rate, the smaller steps taken to reach the minimum error in output. Default is 0.0003
	 `python train.py data_directory_path --learning_rate 0.0005`
- Option 4 is the size of the first hidden layer. Default is 500, and generally should be bigger than 102 but smaller than the input size of the model you want. 
	 `python train.py data_directory_path --hidden_units 600`
- Option 5 is the number of epochs, or the number of times you want to run through all the training data to update it. Default is 2, and note that too many epochs could lead to overfitting (but too little could lead to underfitting..)
	 `python train.py data_directory_path --epochs 5`
- Option 6 is if you would like to run this program on an external gpu (graphic processing unit). This is an option because this program is very large and has many nodes in input and hidden layer(s), so both time and memory are a concern on your computer’s cpu (central processing unit). However, if you do want to run it on an external gpu, you need to connect your computer to a gpu first. Default option is false/ on cpu.
	 `python train.py data_directory_path --gpu true`

2)	Prediction:
The predict_helper.py script contains all the helper functions needed to predict the class of a given image in predict.py. Again, the command line takes multiple arguments:

- First, the required arguments include both the path to the image we want to classify, and the path to the checkpoint with saved model features:
	 `python predict.py image_path checkpoint_path`
- Option 1 is the number of top classes you want to print out. Default is 5:
	 `python predict.py image_path checkpoint_path --top_k 3`
- Option 2 is the path of the json file containing conversions from numerical class title (as the folders are labeled) to categorical class title (actual name of flowers in this case). Default is ‘cat_to_name.json’:
	 `python predict.py image_path checkpoint_path --category_names ‘cat_to_name.json’`
- Option 3 is if you want to use gpu or not (see last option in training above for more info). Default is False:
	 `python predict.py image_path checkpoint_path --gpu true`

## OUTPUT
-	**Train.py** is expected to output train data error/loss, train data accuracy, and validation data accuracy for every 30 batches in each epoch. Then, it is expected to save the trained model features in a folder to the path that you specified, or ‘checkpoint.pth’ by default
-	**Predict.py** is expected to output the top k classes that your image is expected to be, as well as the probabilities for them 




