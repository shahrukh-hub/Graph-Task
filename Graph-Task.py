import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df2= pd.read_csv('Iris_1.csv')
#=================Sepal-Length and species=====================
x= df2['SepalLengthCm']
y= df2['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#============================Bar-Graph=========================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.bar(x,y,color="Gray")
plt.title("Iris Data == Sepal-Length == Bar-Graph")
plt.show()


#=========================Scatter-Graph===========================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.scatter(x,y,color="Red")
plt.title("Iris Data == Sepal-Length == Scatter-Graph")
plt.show()


#============================Line-Graph==============================

plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.plot(x,y,color="Green")
plt.title("Iris Data == Sepal-Length == Line-Graph")
plt.show()

#=========================Sepal-Width and species====================
x= df2['SepalWidthCm']
y= df2['Species']
#=================================Bar-Graph============================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.bar(x,y,color="lawngreen")
plt.title("Iris Data == Sepal-Width ==Bar-Graph")
plt.show()


#==================================Scatter graph============================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.scatter(x,y,color="tomato")
plt.title("Iris Data == Sepal-Width ==Scatter-Graph")
plt.show()


#=====================================Line graph===============================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.plot(x,y,color="gold")
plt.title("Iris Data == Sepal-Width ==Line-Graph")
plt.show()


#==================================Petal-length and species===================

x= df2['PetalLengthCm']
y= df2['Species']

#===============================Bar-Graph===============================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.bar(x,y,color="crimson")
plt.title("Iris Data == Petal-Length ==Bar-Graph")
plt.show()


#==============================Scatter-Graph==============================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.scatter(x,y,color="dodgerblue")
plt.title("Iris Data == Petal-Length ==Scatter-Graph")
plt.show()


#--------------------------------------Line-Graph------------------------
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.plot(x,y,color="coral")
plt.title("Iris Data == Petal-Length ==Line-Graph")
plt.show()

#============================Petal-Width and spscies=====================
x= df2['PetalWidthCm']
y= df2['Species']


#===============================Bar-Graph===============================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.bar(x,y,color="dimgray")
plt.title("Iris Data == Petal-Width ==Bar-Graph")
plt.show()

#==============================Scatter-Graph========================

plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.scatter(x,y,color="lime")
plt.title("Iris Data == Petal-Width =Scatter-Graph")
plt.show()


#================================Line-Graph===========================
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.plot(x,y,color="orchid")
plt.title("Iris Data == Petal-Width==Line-Graph")
plt.show()