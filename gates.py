#orgate
def activate(x):
    return 1 if x>=0 else 0

def perceptron(inputs):
    w1,w2,b=0.1,0.1,-1
    desired_outputs=[0,1,1,1]#[1,0,0,0]#[0,0,0,1],#[1,1,1,0]
    learning_rate=0.1 
    epochs=100
    for epoch in range(epochs):
        total_error=0
        for i in range(len(inputs)):
            A,B=inputs[i]
            target_output=desired_outputs[i]
            output=activate(w1*A+w2*B+b)
            error=target_output-output
            w1+=learning_rate*error*A
            w2+=learning_rate*error*B
            b+=learning_rate*error
            total_error+=abs(error)
        if total_error==0:
            break
    return w1,w2,b
inputs=[[0,0],[0,1],[1,0],[1,1]]
w1,w2,b=perceptron(inputs)
print("weights:",w1,w2)
print("bias:",b)
for i in range(len(inputs)):
    A,B=inputs[i]
    output=activate(w1*A+w2*B+b)
    print("Input: ",A,B,"Output: ",output)
    

    


