#Statistical Machine Learning
#Author: Yongbaek Cho
# Naive Bayes Classifier and Logistic Regression

import scipy.io
import numpy as np

#Naive Bayes Classifier
def read_data():
    Numpyfile = scipy.io.loadmat('mnist_data.mat')#read the MNIST data
    return Numpyfile

def feature_extract(Numpyfile):#he function for extracting features
    complete = [] 
    for a in Numpyfile:
        features = []
        features.append(np.mean(a))#extract the mean(feature)
        features.append(np.std(a))#extract the standard deviation(feature)
        complete.append(features)# add features(mean, std) in the empty list
    return complete

def naiveBayesClassifier(x, y, mu7, std7, mu7_2, std7_2, mu8, std8, mu8_2, std8_2, priorprob_7, priorprob_8):
    prob7_1 = (1/(np.sqrt(2 * np.pi * (std7 ** 2)))) * np.exp(-((x - mu7) ** 2)/(2 * (std7 ** 2))) #normal distribution equation
    prob7_2 = (1/(np.sqrt(2 * np.pi * (std7_2 ** 2)))) * np.exp(-((y - mu7_2) ** 2)/(2 * (std7_2 ** 2))) 
    prob8_1 = (1/(np.sqrt(2 * np.pi * (std8 ** 2)))) * np.exp(-((x - mu8) ** 2)/(2 * (std8 ** 2))) 
    prob8_2 = (1/(np.sqrt(2 * np.pi * (std8_2 ** 2)))) * np.exp(-((y - mu8_2) ** 2)/(2 * (std8_2 ** 2))) 
    if (prob7_1 * priorprob_7) * (prob7_2 * priorprob_7) > (prob8_1 *priorprob_8) * (prob8_2 * priorprob_8):
        return 0
    elif (prob7_1 * priorprob_7) * (prob7_2 * priorprob_7) < (prob8_1 *priorprob_8) * (prob8_2 * priorprob_8):
        return 1
    
def main():
    pre_data = read_data()
    train = pre_data['trX']
    train_ = pre_data['trY']
    test = pre_data['tsX']
    test_ = pre_data['tsY']
    train7, train8, test7, test8 =[], [], [], []
    
    #digit 7 and 8 sperating
    for i in range(len(train_[0])):
        if train_[0][i] == 0:
            train7.append(train[i])
        elif train_[0][i] == 1:
            train8.append(train[i])
    for i in range(len(test_[0])):
        if test_[0][i] == 0:
            test7.append(test[i])
        elif test_[0][i] == 1:
            test8.append(test[i])
  
    train_7 = feature_extract(train7)#extract the features
    train_8 = feature_extract(train8)
       
    mean7, mean8, sd7, sd8 = [],[],[],[]
    for i in range(len(train7)):
        mean7.append(train_7[i][0])
        sd7.append(train_7[i][1])
    for i in range(len(train_8)):
        mean8.append(train_8[i][0])
        sd8.append(train_8[i][1])
    #prior probability
    priorprob_7 = len(train7)/(len(train7) + len(train8))
    priorprob_8 = len(train8)/(len(train7) + len(train8))
    #calculate the parameters for using normal distribution
    #parameters for digits
    mu7 = np.ravel(mean7).mean()
    std7 = np.ravel(mean7).std()
    mu7_2 = np.ravel(sd7).mean()
    std7_2 = np.ravel(sd7).std()
    mu8 = np.ravel(mean8).mean()
    std8 = np.ravel(mean8).std()
    mu8_2 = np.ravel(sd8).mean()
    std8_2 = np.ravel(sd8).std()
        
    #prediction for digit 7
    list_7= []
    for pred in np.array(test7):
        list_7.append(naiveBayesClassifier(pred.mean(), pred.std(), mu7, std7, mu7_2, std7_2, mu8, std8, mu8_2, std8_2, priorprob_7, priorprob_8))
    #prediction for digit 8
    list_8 = []
    for pred in np.array(test8):
        list_8.append(naiveBayesClassifier(pred.mean(), pred.std(), mu7, std7, mu7_2, std7_2, mu8, std8, mu8_2, std8_2, priorprob_7, priorprob_8))
    #accuracy
    accuracy7, accuracy71 = 0 , 0
    for digit in list_7:
        if digit == 0:
            accuracy7 += 1
    for digit in list_8:
        if digit == 1:
            accuracy71 += 1
    
    print("Naive Bayes Classifier- Accuracy for digit 7 is " + str((accuracy7/len(list_7) * 100)) + "%")
    print("Naive Bayes Classifier- Accuracy for digit 8 is " + str((accuracy71/len(list_8) * 100)) + "%")
    print("Naive Bayes Classifier- Total Accuaracy for both 7 and 8 is " + str(((accuracy7 + accuracy71)/ len(list_7+list_8) * 100)) + "%")
main()

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
# Logistic Regression 

def read_data(): 
    Numpyfile = scipy.io.loadmat('mnist_data.mat') #read the MNIST data
   
    return Numpyfile

def feature_extract(Numpyfile): #the function for extracting features
    complete = [] 
    for a in Numpyfile:
        features = []
        features.append(np.mean(a)) #extract the mean(feature)
        features.append(np.std(a)) #extract the standard deviation(feature)
        complete.append(features) # add features(mean, std) in the empty list
    complete = np.array(complete) #convert feature list to array(because of Transpose)
    return complete

def sigmoid(X, W): #sigmoid function
    return 1 / (1 + np.exp(-np.dot(X, W)))

def GradientAlgo(X, Y, predY): #Gradient Algorithm
    return np.dot(X.T, (Y - predY))

def logistic_regression(X, Y,  learningrate , steps): #Logistic Regression function
    weights = [0,0] #initialize weights
    for a in range(steps): # updating weights with gradient ascent algorithms in iteration.
        weights += learningrate * GradientAlgo(X, Y, sigmoid(X, weights))
    return weights

def main():
    pre_data = read_data()
    train = pre_data['trX']
    train_ = pre_data['trY']
    test = pre_data['tsX']
    test_ = pre_data['tsY']
    
    predictions = sigmoid(feature_extract(test),  logistic_regression(feature_extract(train), train_[0],0.005, 3000)) #train and test logistic regression
       
    co1, co2, result1, result2 = 0,0,0,0
    for i in range(len(predictions)):
        if predictions[i] < 0.5: #if the probability of prediction is less than 0.5
            predict_label = 0 #the digit is 7
        elif predictions[i] > 0.5: # otherwise
            predict_label = 1 # the digit is 8
        if predict_label == test_[0][i]:
            if predict_label == 0:
                co1 +=1
            if predict_label == 1:
                co2 += 1
        if test_[0][i] == 0:
            result1 += 1
        elif test_[0][i] == 1:
            result2 +=1
       
    print("Logistic Regression- Accuracy for digit 7 :" + str((co1 / result1) * 100) + "%")
    print("Logistic Regression- Accuracy for digit 8 : " + str(co2/ result2 * 100) + "%")
    print("Logistic Regression- Total Accuracy for both 7 and 8 :" + str(((co1 + co2)/(result1 +result2)) * 100) + "%")
    
main()