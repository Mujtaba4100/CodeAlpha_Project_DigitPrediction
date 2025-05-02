


Project:Digit Prediction

The project consists of three main components:
    1. Model Training (model.py)
    2. Prediction and Evaluation (predictions.py)
    3. Interactive GUI (gui.py)

1. Data
    • Source: train.csv and test.csv(from kaggle)
    • Format: 28x28 grayscale images flattened into 784 pixels
    • Labels: 0–9 (digits)

2. Execution
      In order to excute 1st make sure that train.csv and test.csv paths are correct. Then u can simply paste my model trained named as (Model_Predict_Digit.keras ) or another model(model.keras) in your repository and then run gui.py. A 200x200 window appears with a white screen and 2 options:
    • predict
    • clear
       Steps:
    1. write the digit in white screen make sure its readable and then click predict. Output will be displayed as Predicted digit:X.  
    2.  In order to make another prediction click clear and then rewrite the digit and follow process again.
3. Model Training (model.py)
    • Optimizer: ADAM
    • Loss: Sparse-categorical entropy
    • epoch size: 10
    • saved model: model.keras


4. Prediction (predictions.py)
    • Model : model.keras
    • train test split =80:20
    • epochs:5
    • output saved in “predictions.csv”
    • accuracy:99.39%
5. gui_app(gui.py)
    • Built using tkinter
    • user draws digit in screen
    • image of drawn digit is saved 
    • saved image is centered using center of mass and resized 28x28
    • predicted using trained model
    • out displayed as predicted digit:x
