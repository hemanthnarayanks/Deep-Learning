def activate(x):
    return 1 if x>=0 else 0
def perceptron(inputs):
    target_output=[1,0]
    w,b=0,-1
    epochs=100
    learning_rate=0.1
    for i in range(epochs):
        total_error=0
        
        for i in range(len(inputs)):
            a=inputs[i]
            output=activate(w*a+b)
            error=target_output[i]-output
            w+=error*learning_rate*a
            b+=error*learning_rate
            total_error+=abs(error)
        if(total_error==0):
            break
    return w,b
inputs=[0,1]
w,b=perceptron(inputs)
for i in range(len(inputs)):
    a=inputs[i]
    output=activate(w*a+b)
    print("input: ",a,"Output: ",output)
    

            

            
    

