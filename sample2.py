import numpy as np


#sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))

#Gradient of sigmoid function    
def _sigmoid_derivative(x):
	return x*(1-x)


#input dataset
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

#output dataset                
y = np.array([[0],
			[1],
			[1],
			[0]])

# Seed the random number generator, so it generates the same number every time the program runs
np.random.seed(1)

# randomly initialize our weights with mean 0
w1 = 2*np.random.random((3,4)) - 1
w2 = 2*np.random.random((4,4)) - 1
w3 = 2*np.random.random((4,1)) - 1


def predict(inputs):
        # Pass inputs through our neural network (our single neuron)
    synaptic_weights = [w1, w2, w3]
    i1 = inputs
    for j in range(3):
        i1 = sigmoid(np.dot(i1, synaptic_weights[j]))
    return i1


#training with alpha
#have set the value based on trail-and-error
alpha = 18

for i in xrange(50000):

	# Feed forward through layers 0, 1, 2 and 3
    l0 = X
    l1 = sigmoid(np.dot(l0,w1))
    l2 = sigmoid(np.dot(l1,w2))
    l3 = sigmoid(np.dot(l2,w3))
    
    # how much did we miss the target value?
    l3_error = y - l3
    
    if (i% 5000) == 0:
        print "Error:" + str(np.mean(np.abs(l3_error)))
        
    # in what direction is the target value?
    # are we really sure? 
    l3_delta = l3_error*_sigmoid_derivative(l3)

    # how much did each l2 value contribute to the l3 error (according to the weights)?
    l2_error = l3_delta.dot(w3.T)
    
    # in what direction is the target l2?
    # are we really sure? 
    l2_delta = l2_error * _sigmoid_derivative(l2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(w2.T)

    # in what direction is the target l1?
    # are we really sure?
    l1_delta = l1_error * _sigmoid_derivative(l1)

    w3 += alpha*l2.T.dot(l3_delta)
    w2 += alpha*l1.T.dot(l2_delta)
    w1 += alpha*l0.T.dot(l1_delta)

print "w1:", w1
print "w2:", w2
print "w3:", w3
print "prediction:"
print predict(np.array([[1,1,1]]))