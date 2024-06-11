import numpy as np
import tensorflow as tf
from tensorflow.keras import models,datasets,layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

(X_train,y_train),(X_test,y_test)=datasets.mnist.load_data()

X_train.shape
X_test.shape

X_train=tf.keras.utils.normalize(X_train,axis=1)
X_test_temp=X_test
X_test=tf.keras.utils.normalize(X_test,axis=1)

def image(X,y,ind):
    plt.imshow(X[ind])
    print(y[ind])
image(X_train,y_train,8)

model=models.Sequential([
    layers.Conv2D(input_shape=(28,28,1),kernel_size=(3,3),filters=28,activation='ReLU'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(kernel_size=(3,3),filters=28,activation='ReLU'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(32,activation='ReLU'),
    layers.Dense(10,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10)

y_pred=model.predict(X_test)
y_pred[:5]
y_pred_norm=[np.argmax(element) for element in y_pred]
y_pred_norm[:5]
for new,og in zip(y_pred_norm[:5],y_test[:5]):
    print(new,"\t",og)
    
print(classification_report(y_test,y_pred_norm))

new=np.invert(np.array([X_test_temp[15]]))
image(X_test,y_test,15)
model.predict(new)