import numpy as np
 
epochs = 80000                                  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 2
L = 0.1                                         # learning rate      
 
X = np.array([[0,0], [0,0.5], [0.5,0], [0.5,0.5]])      # Buttons states array
Y = np.array([ [0,1],   [0.5,0.4],   [0.3,1],   [0.2,1]])       # LED states array
 
def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_deriv(x):return x * (1 - x) 
                                                # weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))
 
for i in range(epochs):
 
    H = sigmoid(np.dot(X, Wh))                  # calculate forward part
    Z = np.dot(H,Wz)                            # 
    E = Y - Z                                   # calculate error
    dZ = E * L                                  # delta Z
    Wz +=  H.T.dot(dZ)                          # calculate backpropagation part
    dH = dZ.dot(Wz.T) * sigmoid_deriv(H)        # 
    Wh +=  X.T.dot(dH)                          # update hidden layer weights


Zr= np.around(Z,2)
print("**************** error ****************") 
print(E)
print("***************** output **************") 
print(Z)   
print("***************** output redondeo **************") 
print(Zr)   
print("*************** weights ***************") 
print("input to hidden layer weights: ")     
print(Wh)
print("hidden to output layer weights: ")
print(Wz)
print("con los pesos calculamos para x=[0.5,0]: ")
X=[0.0,0.5]
Hs = sigmoid(np.dot(X, Wh))
print(Hs)
Zs = np.dot(Hs,Wz) 
print("resultado esperado 0.5 , 0.4 : ")
print(Zs)
