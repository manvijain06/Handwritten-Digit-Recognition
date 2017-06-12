# !!!!!! ----------------------------------MANVI JAIN--------------------------------------------------!!!!!
# -------------------COMPUTATIONAL LEARNING AND DISCOVERY : MIDTERM PROJECT----------------------------------
# -------------------ARTIFICIAL NEURAL NETWORK FOR HANDWRITTEN CHARACTER RECOGNITION-------------------------

import numpy as np
import random
import math
import sys


def clean(filename):
#cleaning the data 
    fd =open(filename)
    data=fd.read()
    data=data.strip('\n\t ')
    data=data.replace(' ','')
    data=data.replace('\t','')
    data=data.replace('\r','')
    data=data.split('\n')
    data = [line for line in data if line !='']
    return data

def prep_train_data(name):
#reading the training data files
#appending target values to each line in the file
    list_data =[]
    for i in range(0,10):
        filename = name + str(i) + ".txt"  
        data = clean(filename)
        label = [0.1]*10
        label[i]=0.9
        label = list(label)
        for features in data:
            features = list(features)
            features = [ int(j) for j in features]
            features.insert(0,1) #bias input
            label = [ float(k) for k in label]
            list_data.insert(len(list_data),[features,label])
        
    return list_data
        
def prep_test_data(name):
#reading the testing data files
#appending target values to each line in the file
    list_data =[]
    for i in range(0,10):
        filename = name + str(i) + ".txt"  
        data = clean(filename)
        label = [0.1]*10
        label[i]=0.9
        label = list(label)
        for features in data:
            features = list(features)
            features = [ int(j) for j in features]
            features.insert(0,1) #bias input
            label = [float(k) for k in label]
            list_data.insert(len(list_data),[features,label])
            
    return list_data

def shuffle( data ):
    # Shuffle the data within a training set to get better weight learning updates
    # This function is used in the training phase
    for i in range(len(data)):
        index = random.randint(0,len(data)-1)
        tmp = data[index]
        data[index] = data[i]
        data[i] = tmp
    return data

def cross_validation(dataset, k_fold, itr ): 
#to divide training dataset into training and validation data
#k_fold : the folds we want for the data, 
#itr: the iteration number of crossvalidation cycle
    l = len(dataset)
    val_chunk = int(math.floor(float(l)/k_fold))
    lower_limit = itr * val_chunk
    upper_limit = lower_limit + val_chunk
    validation_set = dataset[lower_limit:upper_limit]
    train_set = dataset[0: lower_limit + 1]
    train_set.extend(list(dataset[upper_limit :-1]))
    return (train_set, validation_set)
              

def shuffle( data ):
    # Shuffle the data within a training set to get better weight learning updates
    # This function is used in the training phase
    for i in range(len(data)):
        index = random.randint(0,len(data)-1)
        tmp = data[index]
        data[index] = data[i]
        data[i] = tmp
    return data

def sigmoid(x):
    return float(1)/(1+math.exp(-x))

"""def sigmoid_der(x):
    return sigmoid(x)(1-sigmoid(x))"""


def feedforward(inputs,  w_ih, w_ho):
    # w_ih = weights from input layer to hidden layer
    # w_ho = weights from hidden layer to output layer
    # output_ih = hidden layer output
    # output_ho = output of output layer
    #input_ih = input from input to hidden layer
    #input_ho = input from hidden to output layer

    input_ih = np.dot((inputs), np.array(w_ih))
    output_ih = [sigmoid(x) for x in input_ih]
    output_ih.insert(0,1) # Account for bias input 
    input_ho = np.dot(output_ih, np.array(w_ho))
    output_ho = [sigmoid(x) for x in input_ho]
    return (output_ih, output_ho)
    
def backpropagate(inputs, outputs, targets, w_ih, w_ho, old_wts, learningRate , momentum ):
    
    # inputs = 65 units (64 attribute + bias)
    # outputs = 10 units 
    # old_wts = weight update of the previous iteration (used to add momentum)
    
    output_ih, output_ho = outputs
    output_ih = np.array(output_ih)
    output_ho = np.array(output_ho)

    
    targets = np.array(targets) #expected target values from the network
    
    k = len(output_ho)
    ones = np.ones((k,), dtype=np.int)
    op_gradient = np.multiply(output_ho, ones-output_ho)
    outputDelta = np.multiply(op_gradient, targets-output_ho)
    """
    outputDelta = sigmoid_derivative(output_ho)*(targets - output_ho)
    hiddenDelta = sigmoid_derivative(output_ih)*(np.dot(np.array(w_ho),outputDelta.transpose()))"""
    
    k = len(output_ih)
    ones = np.ones((k,), dtype=np.int)
    hidden_gradient = np.multiply(output_ih, ones-output_ih)

    # Calculate downstream sum for all hidden units at once using 
    # matrix operation
    
    downstream_sum = np.dot(np.array(w_ho),outputDelta.transpose())
    hiddenDelta = np.multiply(hidden_gradient, downstream_sum)

    
    l1 = len(np.array(inputs))
    l2 = len(hiddenDelta)

    delta_wih = learningRate*(np.dot(np.array(inputs).reshape(l1,1),hiddenDelta.reshape(1, l2)))

    # Remove bias weights at hidden layer, as it does not contribute to input layer's weight updation
    delta_wih = np.array(delta_wih.transpose().tolist()[1:len(delta_wih)]).transpose()

    # The weight update from input to hidden layer
    wih_new = np.add(np.array(w_ih), delta_wih)

    # Add momentum term
    if old_wts !=None:
         wih_new = np.add(np.array(wih_new), momentum*old_wts[0])

    l1 = len(np.array(output_ih))
    l2 = len(outputDelta)

    delta_who = learningRate*(np.multiply(np.array(output_ih).reshape(l1,1), outputDelta.reshape(1,l2)))

    # The weight update for hidden to output layer
    who_new = np.add(np.array(w_ho), delta_who)

    # Add momentum term
    if old_wts !=None:
         who_new = np.add(np.array(who_new), momentum*old_wts[1])
         
    # Return a tuple of the weights in the two layers
    return [ (wih_new.tolist(), who_new.tolist()), (delta_wih, delta_who)]

def training_of_network(train_set,  hiddenunits, validation_set=None, last_epoch=None, num_outputs=10):
    w_ih = []
    w_ho = []
    learningRate = 0.02 
    momentum = 0.7

    # Initialize weights
    for i in range(len(train_set[0][0])): # training set is already appended with input bias on prepare_data()
        wt = [random.uniform(-0.05,0.05) for i in range(0, hiddenunits)]
        w_ih.insert(len(w_ih),list(wt))

    for i in range(hiddenunits+1): # Adds one more to hidden units to account for hidden layer input bias
        wt = [random.uniform(-0.05,0.05) for i in range(0, num_outputs)]
        w_ho.insert(len(w_ih),list(wt))

    # An epoch is a full run of feedforward/back-propagation for the entire training set 
    epoch = 0

    # least_error obtained while training and min epochs needed to achieve the min error
    least_error = 1
    leastnoofepoch = 0

    present_error = 1


    last_error = 0
    h = 0 # Counts the number of hillations

    # last_error is the weight updates of the previous iteration (used with momentum term)
    old_wts = None

    #STOPPING CRITERIA ( WHEN ERROR < 0.03 AND ERROR OSCILLATION <= 25 AND EPOCH < 2000)
    while  (last_epoch!=None and epoch<=last_epoch) or( last_epoch==None and 
                                                      (present_error - least_error <0.03) and
                                                      ( h <=25) and (epoch <2000)): 
        train_set = shuffle(train_set)
        train_attribute = [x[0] for x in train_set]
        train_targets = [x[1] for x in train_set]

        # Run the stochastic gradient descent for all the training data
        for inp in range(len( train_attribute)):
            outputs = feedforward(train_attribute[inp], w_ih, w_ho)
            new_wts = backpropagate(train_attribute[inp], outputs, train_targets[inp],w_ih, w_ho,
                                          old_wts,learningRate,momentum)
            old_wts = new_wts[1]
            w_ih, w_ho = new_wts[0]


        # VALIDATE NETWORK EVERY 10 EPOCHS(10 K_FOLDS)  
        if last_epoch != None:
            final_wts = new_wts[0]
        elif last_epoch ==None and epoch % 10 ==0:
            present_error = validate(validation_set, new_wts[0])
            
            if abs(present_error-last_error) <= 0.01:
                h+=1
            last_error = present_error
            if present_error <least_error:
                least_error = present_error
                leastnoofepoch = epoch
                h= 0 # hillations are reset when a new minima is found
                final_wts = new_wts[0]
            
        epoch += 1
    return (leastnoofepoch,final_wts) # Return epochs needed to obtain min error and the weights learned for min error

def validate(validation_set,weights):
    # Test the learned weights on validation set
    validate_attribute = [x[0] for x in validation_set]
    validate_targets = [x[1] for x in validation_set]
    w_ih, w_ho = weights

    error = 0
    len(validate_attribute)

   
    # the highest activation among the 10 output units =  training attribute . tie, classify it as an error.
    for inp in range(len(validate_attribute)):
        outputs = feedforward(validate_attribute[inp], w_ih, w_ho)
        (o1, op)=outputs
        t = validate_targets[inp]
        t1 = np.argmax(op)
        t2 = np.argmax(validate_targets[inp])
        same = False
        if (t1 == t2):
            same = True
        elif (t1 != t2):
            error += 1 

    return float(error)/len(validate_attribute)

def testing_of_network(test_set, weights):

    # Test the learned weights on test data
    test_attribute = [x[0] for x in test_set]
    test_targets = [x[1] for x in test_set]
    w_ih, w_ho = weights
    perdigitlen = len(test_set)//10

    errors_lst = []
    sum_errors = 0
    inp = 0
    f_max= 0
    diff_max=0

    
    f_d = [] # List to store the first best activation of the 10 output units, for each digit
    f_d_diff = [] #for second activation

    # highest activation -> training attribute. In case of tie, classified as an error.
    
    for digit in range(10):
        errors = 0
    
        f_d_max =0
        f_d_diff_max=0
        for r in range(perdigitlen):
            outputs = feedforward(test_attribute[inp], w_ih, w_ho)
            (o1,op)=outputs
            t = test_targets[inp]
            t_ind = t.index(max(t))

            match = False
            indices = [ind for ind, val in enumerate(op) if val == max(op)]
            first_max = max(op)
            for ind in indices:
                op[ind]=-1
            sec_max = max(op)
            
    
            f_max+= first_max
            f_d_max +=first_max
            f_d_diff_max+=first_max-sec_max
            diff_max += first_max-sec_max
            if (len(indices) == 1) and (indices[0] == t_ind):
                match=True
            if match == False:
                errors += 1
            u = float(errors/perdigitlen)
            CI =  u + 1.96 *maths.sqrt((u*(1-u))/perdigitlen)
                        
            inp +=1
        
        sum_errors += errors
        
        f_d.append(float(f_d_max)/perdigitlen)
        f_d_diff.append(float(f_d_diff_max)/perdigitlen)
        errors_lst.append(float(errors)/perdigitlen)
        
        

    return (errors_lst, float(sum_errors)/len(test_attribute), 
            float(f_max)/len(test_attribute),float(diff_max)/len(test_attribute),
            f_d, f_d_diff,CI)

def main():
    name_train_file="train"
    name_test_file="test"
    traindata = prep_train_data(name_train_file)
    testdata = prep_test_data(name_test_file)

    # Randomize the data
    train_set = shuffle(traindata)
    
    epochiterations = 0
    
    # Partition the training data into chunks in order to separate a validation set
    k_fold = 6

    hiddenunits = 4

    # Cross-validation of data.
    for chunk in range(k_fold):
        (training_set,validation_set) = cross_validation(train_set, k_fold, chunk)
        print("Size of training_set and validation_set:",(len(training_set), len(validation_set)))
        (epochs, weights) = training_of_network(training_set, hiddenunits, validation_set)
        epochiterations = epochiterations + epochs
        
    epochiterations = epochiterations//k_fold

    #last epoch after cross validation
    (e, weights) = training_of_network(train_set,hiddenunits, last_epoch=epochiterations)
    
    print("The misclassifiers reduced for validation data, hence we run the test data now")

    # Test the network
    w_ih, w_ho = weights
    err_lst, tot_error, f_max, diff_max, f_max_digit, diff_max_digit,confidence_interval = testing_of_network(testdata, weights)
   
    print("The error is")
    for i,e in enumerate(err_lst):
        print i,str(e)
    print "Total error:",(str(tot_error))
    
    print(str(f_max))
   
    print(str(diff_max))
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for e in f_max_digit:
        print(str(e))
        
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i,e in enumerate(diff_max_digit):
        print i ,str(e) 
    
    for i in range(err_lst):
        print i, confidence_interval
    
    return 
   
main()
