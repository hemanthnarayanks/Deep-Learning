import numpy as np
class gates:
    def __init__(self,learning_rate,epochs):
        self.learning_rate=learning_rate
        self.epochs=epochs
        
    def activate(self,x):
        if x>=0:
            return 1
        else:
            return 0
        
    def ORpercep(self,inputs):
        w_or=np.array([0.1,0.1])
        b_or=-1
        target_output=[0,1,1,1]
        for epoch in range(self.epochs):
            total_error=0
            for i in range(len(inputs)):
                a,b=inputs[i]
                x=np.array([a,b])
                output=self.activate(np.dot(w_or,x)+b_or)
                error=target_output[i]-output
                w_or[0]+=self.learning_rate*error*x[0]
                w_or[1]+=self.learning_rate*error*x[1]
                b_or+=self.learning_rate*error
                total_error+=abs(error)
            if total_error==0:
                break
        return w_or,b_or
    
    def ANDpercep(self,inputs):
        w_and=np.array([0.1,0.1])
        b_and=-1
        target_output=[0,0,0,1]
        for epoch in range(self.epochs):
            total_error=0
            for i in range(len(inputs)):
                a,b=inputs[i]
                x=np.array([a,b])
                output=self.activate(np.dot(w_and,x)+b_and)
                error=target_output[i]-output
                w_and[0]+=self.learning_rate*error*x[0]
                w_and[1]+=self.learning_rate*error*x[1]
                b_and+=self.learning_rate*error
                total_error+=abs(error)
            if total_error==0:
                break
        return w_and,b_and
    
    def NOTperceptron(self,inputs):
        w_not=1
        b_not=-1
        target_output=[1,0]
        for epoch in range(self.epochs):
            total_error=0
            for i in range(len(inputs)):
                x=inputs[i]
                output=self.activate(w_not*x+b_not)
                error=target_output[i]-output
                w_not+=self.learning_rate*error*x
                b_not+=self.learning_rate*error
                total_error+=abs(error)
            if total_error==0:
                break
        return w_not,b_not
    
xor=gates(0.1,100)

inputs_or_and = [[0,0],[0,1],[1,0],[1,1]]

inputs_not = [0,1]

w_and,b_and = xor.ANDpercep(inputs_or_and)
w_or,b_or = xor.ORpercep(inputs_or_and)
w_not,b_not = xor.NOTperceptron(inputs_not)
print("Inputs:\tOR\tAND\tNAND\tXOR")
for i in range(len(inputs_or_and)):
    y1 = xor.activate(np.dot(w_or,inputs_or_and[i])+b_or)
    y2 = xor.activate(np.dot(w_and,inputs_or_and[i])+b_and)
    y3 = xor.activate(w_not*y2+b_not)
    values = np.array([y1,y3])
    y4 = xor.activate(np.dot(w_and,values)+b_and)
    print(inputs_or_and[i],y1,y2,y3,y4)
    
    

                
            
    
        
        
    

