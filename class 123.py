import cv2
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os
import ssl
import time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl,'_create_unverified_context',None)):
        ssl._create_default_https_context= ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=2500, train_size=7500, random_state=9)
x_train = x_train/255
x_test = x_test/255

model = LogisticRegression(solver='saga', # libilinear is for linear logistic regression multi_class='multinomial', # labels 0 to 1 and best solver is saga )
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    

    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0),2)  # imaage,w,h,color,thickness

    #roi = Region Of Interest
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #convert the imge from ay format to pil format, so using python we can manipulate the image and predict
    im_pil = Image.fromarray(roi)  # create a image obj from roi

    # convert graycale image - L format, each pixel represnted btwn 0 to 100
    image_bw = im_pil.convert('L')

    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)  # resize image

    #invert the image
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

    # gt the minimum pixel in image
    min_pixel = np.percentile(image_bw_resized_inverted, 20)

    #clip.. used to limit the pixels betwen 0 and 255
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - min_pixel, 0, 255)

    # now get the max pixel present in the image
    max_pixel = np.max(image_bw_resized_inverted)

    #change the image into arrays, to predict the values
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

    #creating a test sample andmaking a prediction
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
    test_pred =model.predict(test_sample)
    print("Predicted class is: ",test_pred)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()