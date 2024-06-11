#adaline
import numpy as np
class Adaline:
    def __init__(self,learning_rate,epochs,error):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.error=error
    
    def network(self,inputs):
        w=np.array([0.1,0.1])
        bias=0.1
        targets=[-1,1,1,1]
        for epoch in range(self.epochs):
            total_error=0
            for i in range(len(inputs)):
                x=np.array(inputs[i])
                output=np.dot(w,x)+bias
                w[0]+=self.learning_rate*x[0]*(targets[i]-output)
                w[0]+=self.learning_rate*x[0]*(targets[i]-output)
                bias+=self.learning_rate*(targets[i]-output)
                local_error=(targets[i]-output)**2
                total_error+=local_error
            if(total_error<self.error):
                break
        return w,bias
orgate=Adaline(0.1,100,0.9)
inputs=[[-1,-1],[-1,1],[1,-1],[1,1]]

w,bias=orgate.network(inputs)

for i in range(len(inputs)):
    print(inputs[i],"\t",np.dot(w,np.array(inputs[i]))+bias)
            
                

