# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Thilak Raj . P
RegisterNumber:  212224040353
``` 
```
import pandas as pd
data=pd.read_csv(r"Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## HEAD
![image](https://github.com/user-attachments/assets/97184c2f-10f9-410d-9a67-9a1ae045c0eb)

## COPY
![image](https://github.com/user-attachments/assets/066e0dee-eb3b-4680-8a3d-818d23512c5c)

## FIT TRANSFORM
![image](https://github.com/user-attachments/assets/f61b8e7d-f619-4234-b408-43fbb7604288)

## LOGISTIC REGRESSION
![image](https://github.com/user-attachments/assets/dc914706-a024-4cbc-a89f-d9eb000826c0)

## ACCURACY SCORE
![image](https://github.com/user-attachments/assets/11c18b6f-8782-42b7-b2f3-9bae3a883dfa)

## CONFUSION MATRIX
![image](https://github.com/user-attachments/assets/64dec349-6f4d-4716-9284-7ff64fb7cddf)

## CLASSIFICATION REPORT
![image](https://github.com/user-attachments/assets/7e7bb026-9ba5-4067-b5f1-e2bc0c401b5c)

## PREDICTION
![image](https://github.com/user-attachments/assets/afb9d08c-6dea-4697-b78a-19b0b114139b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
