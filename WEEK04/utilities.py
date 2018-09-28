# coding=utf-8

import numpy
import pylab

### 
# Calculates the Minkowski distance between a vector x and a matrix Y
def minkowski_mat(x,Y,p=2):
    return (numpy.sum((numpy.abs(x-Y))**p,axis=1))**(1.0/p)


##
# The function tests takes as input:
#   labelsTest - the test labels
#   labelsPred - the predicted labels
# and return a table presenting the results
###
def teste(labelsTest, labelsPred):

	n_classes = max(labelsTest)
	conf_matrix = numpy.zeros((n_classes,n_classes))

	for (test,pred) in zip(labelsTest, labelsPred):
		conf_matrix[test-1,pred-1] += 1

	return conf_matrix
	

# fonction plot
def gridplot(classifier,train,test,n_points=50):

    train_test = numpy.vstack((train,test))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))

    xgrid = numpy.linspace(min_x1,max_x1,num=n_points)
    ygrid = numpy.linspace(min_x2,max_x2,num=n_points)

	# calculates the Cartesian product between two lists
    # and puts the results in an array
    thegrid = numpy.array(combine(xgrid,ygrid))

    the_accounts = classifier.compute_predictions(thegrid)
    classesPred = numpy.argmax(the_accounts,axis=1)+1

    # The grid
    # So that the grid is prettier
    #props = dict( alpha=0.3, edgecolors='none' )
    pylab.scatter(thegrid[:,0],thegrid[:,1],c = classesPred, s=50)
	# The training points
    pylab.scatter(train[:,0], train[:,1], c = train[:,-1], marker = 'v', s=150)
    # The test points
    pylab.scatter(test[:,0], test[:,1], c = test[:,-1], marker = 's', s=150)

    ## A little hack, because the functionality is missing at pylab ...
    h1 = pylab.plot([min_x1], [min_x2], marker='o', c = 'w',ms=5) 
    h2 = pylab.plot([min_x1], [min_x2], marker='v', c = 'w',ms=5) 
    h3 = pylab.plot([min_x1], [min_x2], marker='s', c = 'w',ms=5) 
    handles = [h1,h2,h3]
    ## end of the hack

    labels = ['grille','train','test']
    pylab.legend(handles,labels)

    pylab.axis('equal')
    pylab.show()

## http://code.activestate.com/recipes/302478/
def combine(*seqin):
    '''returns a list of all combinations of argument sequences.
for example: combine((1,2),(3,4)) returns
[[1, 3], [1, 4], [2, 3], [2, 4]]'''
    def rloop(seqin,listout,comb):
        '''recursive looping function'''
        if seqin:                       # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]     # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:],listout,newcomb)
        else:                           # processing last sequence
            listout.append(comb)        # comb finished, add to list
    listout=[]                      # listout initialization
    rloop(seqin,listout,[])         # start recursive process
    return listout

