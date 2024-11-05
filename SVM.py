import pandas as pd
import numpy as np

df=pd.read_csv("C:\\Users\\nitin\\Desktop\\Machine Learning\\Social_Network_Ads.csv")

df.head()

df.shape

x=df.iloc[:, [0,1]].values
y=df.iloc[:, 2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=3)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=2)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
disp.plot()

from sklearn.metrics import accuracy_score
print("accuracy: ", accuracy_score(y_test,y_pred))

classifier.score(x_test,y_test)



