import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import datasets,models,layers

(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()

print(X_train.shape)
print(X_test.shape)

with open('batches.meta','rb') as f:
    data=pickle.load(f)
label_names=data['label_names']
print(label_names)

def image(X,y,ind):
    plt.figure(figsize=(10,5))
    plt.imshow(X[ind])
    print(label_names[y[ind]])

image(X_train,y_train,3)

label_names[0]

y_train.shape
y_train=y_train.reshape(-1,)#2d to 1d
label_names[y_train[7]]

image(X_train,y_train,8)

X_train=X_train/255
X_test=X_test/255

model=models.Sequential([
    layers.Conv2D(input_shape=(32,32,3),kernel_size=(3,3),filters=32,activation="ReLU"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(kernel_size=(3,3),filters=64,activation="ReLU"),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64,activation="ReLU"),
    layers.Dense(10,activation="softmax")
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10)

y_test[:5]
y_test.reshape(-1,)
y_pred=model.predict(X_test)
y_pred[:5]

y_pred_norm=[np.argmax(element) for element in y_pred]
y_pred_norm[:5]

for pred,real in zip(y_pred_norm[:10],y_test[:10]):
    print(label_names[pred],"\t",label_names[real[0]])
    
print(classification_report(y_test, y_pred_norm))

image=np.invert(np.array([X_test[0]])) 
np.argmax(model.predict(image))
image(X_test,y_test,0)
