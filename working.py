import numpy as np

def print_header(text):
    print("*" * 80)
    print(text.upper())
    print("*" * 80)

np.random.seed(1)

def relu(x):
    # return x if x is positive otherwise zero
    return (x > 0) * x

def relu2deriv(x):
    # return 1 if x >1 else 0
    return x > 0

##########################
print_header("dataset")
##########################
streetlights = np.array([[0, 1, 1],
                         [1, 1, 0],
                         [0, 0, 0],
                         [1, 1, 1]])
print("streetlights: %s" % (streetlights))

walk_no_walk = np.array([[1,1,0,1]]).T #.T
print("walk_no_walk %s" % (walk_no_walk))

##########################
print_header("parameters")
##########################

hidden_neurons = 4
alpha = 0.1
print(f"hidden_neurons: {hidden_neurons}")
print(f"alpha: {alpha}")
weights_0_1 = np.random.random((len(streetlights[0]), hidden_neurons)) * 2 - 1
print(weights_0_1.shape)
print("weights_0_1: %s" % (weights_0_1))
weights_1_2 = np.random.random((hidden_neurons, 1)) * 2 - 1
print("weights_1_2: %s" % (weights_1_2))


##########################
print_header("inference test")
##########################

layer_1 = streetlights[0:1]
print("layer_1: %s" % (layer_1))
layer_2 = np.dot(layer_1, weights_0_1)
print(f"layer_2 BEFORE RELU: {layer_2}")
layer_2 = relu(np.dot(layer_1, weights_0_1))
print("layer_2: %s" % (layer_2))
layer_3 = np.dot(layer_2, weights_1_2)
print("layer_3: %s" % (layer_3))

delta = layer_3 - walk_no_walk[0:1]
print(f"delta: {delta}")
error = delta ** 2
print(f"error: {error}")


##########################
print_header("training")
##########################

runs = 600 #60
# for each run
for count in range(runs):
    # for each training example
    for idx in range(len(streetlights)):
    #for idx in [0]:
        layer_1 = streetlights[idx:idx+1]
        #print("layer_1: %s" % (layer_1))
        layer_2 = relu(np.dot(layer_1, weights_0_1))
        #print("layer_2: %s" % (layer_2))
        layer_3 = np.dot(layer_2, weights_1_2)
        #print("layer_3: %s" % (layer_3))
        error = (layer_3 - walk_no_walk[idx:idx+1]) ** 2 # ?
        #print("error: %s" % (error))
        layer_3_delta = (layer_3 - walk_no_walk[idx:idx+1])
        #print("layer_3_delta: %s" % (layer_3_delta))
        # GOOD TO THIS POINT
        # GOOD TO THIS POINT
        # GOOD TO THIS POINT

        # we get the deltas for l3 and l2 first.
        layer_2_delta = layer_3_delta.dot(weights_1_2.T) * relu2deriv(layer_2)
        #print("layer_2_delta: %s" % (layer_2_delta))

        #finally adjust weights!
        weights_1_2 -= layer_2.T.dot(layer_3_delta) * alpha
        weights_0_1 -= layer_1.T.dot(layer_2_delta) * alpha

    if (count % 10 == 9):
        print(f"ERROR: {error}")

        #layer_3_deriv = layer_3_delta.dot(weights_1_2.T)
        #print("layer_3_deriv: %s" % (layer_3_deriv))
        #
        #weights_1_2 = weights_1_2 + (layer_3_deriv * alpha)
        # might be right?

        #layer_2_deriv = layer_3_deriv.dot(weights_0_1.T)
        #print("layer_2_deriv: %s" % (layer_2_deriv))
