# Emotion-theme-recognition-app

An interactive web application which could recognize music emotion and themes using one of three pretrained models and return the probability in percentages.

## First steps

- Clone this repo 
- Install requirements
- Download the three directories with pretrained models from this Google Drive **[link](https://drive.google.com/drive/folders/1lwLStaXTG9h9-HDmB9gKjwCumiOWNiqY?usp=sharing)** and place them all to the models directory.
- Run the script passing appropriate arguments
<p align="center">
  <img src="https://user-images.githubusercontent.com/31742528/123638983-6e8da480-d828-11eb-9b87-85cd8335d7a9.png" height="300px" alt="">
</p>

- Go to http://localhost:5003

:point_down: Screenshot:
<p align="center">
  <img src="https://user-images.githubusercontent.com/31742528/123639697-215e0280-d829-11eb-8c89-355b1446afbf.png" height="420px" alt="">
</p>

## Models
You could choose which model you want to use at first. There are three accessable variants of models: ResNet50, InceptionNetV2, EfficientNetB3. Implementations of preprocessing stage is located in repository **[Emotion-and-Theme-Recognition-in-Music-Task](https://github.com/JuliaGerasymenko/Emotion-and-Theme-Recognition-in-Music-Task)** and training the model stage - in `*.py` files with hold the names of the available models, located in current repository. 
