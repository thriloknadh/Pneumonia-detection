Pneumonia Detection From X-Rays

Introduction:

Nowadays due to advance in deep learning and computer vision to create software to assist medical physicians. In this project, we are using a computer vision(CI) with a convolutional neural network (CNN) model is trained to predict the patient have or not of pneumonia from chest X-Ray images. The VGG16 CNN model was used for this classification task. The purpose of use for this model is to pre-screen chest X-Ray images prior to radiologists' review to reduce their time.
This project is organized in three Jupyter Notebooks:

1_EDA (Exploratory Data Analysis):

NIH X-Ray Dataset and X-ray image pixel-level analysis.

2_Build_and_Train_Model: Image pre-processing with Keras ImageDataGenerator, split dataset using Scikit-Learn, build & train a Keras Sequential model, and convert probabilistic outputs to binary predictions(easy to process).
test1.dcm
3.Test the model giving random images:

Dataset:

This project uses the ChestX-ray14 dataset released by NIH Clinical Center. It is comprised of 74 X-Ray images with disease labels from 20,805 unique patients.
The disease labels for each image were created using Natural Language Processing (NLP) to process associated radiological reports for fourteen common pathologies. The estimated accuracy of the NLP labeling accuracy is estimated to be >90%.

Project Overview

Exploratory Data Analysis
Building and Training Your Model
Inference
FDA Preparation
Part 1: Exploratory Data Analysis

Open 1_EDA.ipynb with Anaconda for exploratory data analysis. The following data are examined:

ChestX-ray14 Dataset metadata contains information for each X-Ray image file, the associated disease findings, patient gender, age, patient position during X-ray, and image shape.
Pixel level assessment of X-Ray image files by graphing Intensity Profiles of normalized image pixels. X-Rays are also displayed using scikit-image.
Part 2: Building and Training Your Model, Fine Tuning Convolutional Neural Network VGG16 for Pneumonia Detection from X-Rays
Inputs:

ChestX-ray dataset containing 74 X-Ray images (.png) in data/images and metadata in data folder.
Output:

CNN model trained to classify a chest X-Ray image for presence or absence of pneumonia containing model weights.

Open 2_Build_and_Train_Model with Jupyter Notebook.

Create training data and validation data splits with scikit-learn train_test_split function.

Ensure training data split is balanced for positive and negative cases. Ensure validation data split has a positive to negative case ratio that reflects clinical scenarios. Also check that each split has demographics that are reflective of the overall dataset.

Prepare image preprocessing for each data split using Keras ImageDataGenerator.

To fine-tune the ImageNet VGG16 model, create a new Keras Sequential model by adding VGG16 model layers and freezing their ImageNet-trained weights. Subsequently add Dense and Dropout layers, which will have their weights trained for classifying chest X-Ray images for pneumonia.

The model training will have a history to show loss metrics at each training epoch. The best model weights are also captured at each training epoch.

Model predictions initially return as probabilities between 0 and 1. These probabilistic results were compared against ground truth labels.

A threshold analysis was completed to select the boundary at which probabilistic results are converted into binary results of either pneumonia presence or absence.
The CheXNet algorithm achieved an F1 score of 0.435, while a panel of four independent Radiologists averaged an F1 score of 0.387 [2]. This project's final F1 score is 0.36, which is similar in performance to the panel of Radiologist.

Prediction VS Actual value:
 At the end of the conclusion we get the accuracy,precion recall of the model then we give new input images to test how the machine works so it gets 70% correct answers.
