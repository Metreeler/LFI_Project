# Handwritten letters recognition by artificial intelligence - LFI project

## 1) Project description
This project is the final assignment of the Learning from Images course of Berliner Hochscule für Technik. 
The main idea is to get an interface which permits the user to crop few rectangles in an image. Once he is done a trained neuronal network will determine each letter the rectangles contain. The output of this application is the printed word that is made of those letters. 
To do so, the code permits to create a neuronal network, train, evaluate and test it. Four different neuronal architectures have been tested during this project and you can also find them in the code. The results of the tests are printed and saved. 

## 2) How to install and run the project 

### Train the model:
- create the `data` folder 
- download the csv here : https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format?resource=download (we did not upload it to the git because the file is too big) 
- save it in the `data` folder 
- use one of the four available models or encode the model architecture you want without forgetting to assign it a name (ex : `model_5`)
- if creating a new model, don't forget to load correctly the data according to the model
- write the parameters you want for the model (learning rate, momentum, ... see the parameters slide of the presentation)
- run the program
- indicate that you want to "train" a model
- enter the name of the model
- wait for the model to be trained
- observe in the "save" folder the evolution of the loss, the accuracy, the model summary and the model save (which can be used to test the model see below)

### Use the project:
- save the image to be tested in the images folder
- run the program
- indicate that you want to "test" the model
- follow the instructions given in the output
- observe the result

## 3) Credits 
The members of this project are Malo Hangran and Simon Molz. 
We gratefully thank Professor Kristian Hildebrand for the lessons throughout the semester and the Master class for its attention during our presentation. 
Th list of the website we visited to complete this project are available at the end of the presentation.
