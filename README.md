# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 . Import the required libraries.

2 . Load the dataset.

3 . Define X and Y array.

4 . Define a function for costFunction,cost and gradient.

5 . Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHIV SUJAN S R
RegisterNumber:  212223040194
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:

## Array Value of x :

![367994836-40e6025f-1a85-4755-8161-28960ea19f2f](https://github.com/user-attachments/assets/fc4f49ec-62fb-4fac-9c6e-e794cce0e584)

## Array Value of y :

![367994915-7f596c60-6d35-4baa-a9d1-9cbc4006e050](https://github.com/user-attachments/assets/cf3c10fd-3578-46b4-86c0-0eb87ccabe28)

## Exam 1 - score graph :

![367995007-248727de-21d8-42aa-88c3-b5b5c72f06c5](https://github.com/user-attachments/assets/185a6b3a-4a2e-4c69-92af-1d59823e28d8)

## Sigmoid function graph :

![367995063-790f098e-b0e7-46e5-b911-6c1a8aa63e6e](https://github.com/user-attachments/assets/05091b87-665f-4377-81e8-0f1bd3c44380)

## X_train_grad value :

![367995115-f269e638-746f-46b0-a1da-7b4a3cb61746](https://github.com/user-attachments/assets/691e075d-0491-4ac5-b87d-0370ba5ffde0)

## Y_train_grad value :

![367995171-512baf4d-7aa8-434e-b8b8-fb597b2225cd](https://github.com/user-attachments/assets/6b9e25f0-830d-44c7-b09e-5f4e128429b5)

## Print res.x :

![367995243-d3b24c97-7d71-43c9-8b3b-976b08cad6be](https://github.com/user-attachments/assets/7ac3b82b-97a0-446f-b271-592a199a67c5)

## Decision boundary - graph for exam score :

![367995333-118107da-e2be-4bc8-9dc1-20b8706110ad](https://github.com/user-attachments/assets/8249f450-bc2b-4a8d-a847-ab8b9b2246d3)

## Proability value :

![367995400-f579c894-47af-44f8-a7a4-22d3ab1bb4da](https://github.com/user-attachments/assets/322ab516-ce2f-45e8-89c6-52a608887638)

## Prediction value of mean :

![367995476-4b5b3d15-05ba-4c82-a5a1-c8d360166ad4](https://github.com/user-attachments/assets/2973fbe5-649e-4498-b3c0-58c6c3f707f3)

## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

