# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

# ------------------------ 

# crossval
def crossval(X, Y, n_iterations, iteration):
    Xtest = X[iteration*(len(X)//n_iterations):(iteration+1)*(len(X)//n_iterations)]
    Ytest = Y[iteration*(len(Y)//n_iterations):(iteration+1)*(len(Y)//n_iterations)]
    Xapp = np.concatenate((X[0:iteration*(len(X)//n_iterations)], X[(iteration+1)*(len(X)//n_iterations):len(X)]))
    Yapp = np.concatenate((Y[0:iteration*(len(Y)//n_iterations)], Y[(iteration+1)*(len(Y)//n_iterations):len(Y)]))   
    return Xapp, Yapp, Xtest, Ytest

# crossval_strat
def crossval_strat(X, Y, n_iterations, iteration):
    Y_class = np.unique(Y)
    li_1 = np.where(Y == Y_class[1])
    li_2 = np.where(Y == Y_class[0])
    X1 = X[li_2]
    X2 = X[li_1]
    Y1 = Y[li_2]
    Y2 = Y[li_1]
    Xtest = np.concatenate((X1[iteration*(len(X1)//n_iterations):(iteration+1)*(len(X1)//n_iterations)], X2[iteration*(len(X2)//n_iterations):(iteration+1)*(len(X2)//n_iterations)]))
    Ytest = np.concatenate((Y1[iteration*(len(Y1)//n_iterations):(iteration+1)*(len(Y1)//n_iterations)], Y2[iteration*(len(Y2)//n_iterations):(iteration+1)*(len(Y2)//n_iterations)]))
    L1 = li_2[0][iteration*(len(X1)//n_iterations):(iteration+1)*(len(X1)//n_iterations)]
    L2 = li_1[0][iteration*(len(X2)//n_iterations):(iteration+1)*(len(X2)//n_iterations)]
    L = np.concatenate((L1, L2))
    Xapp = np.delete(X, L,0)
    Yapp = np.delete(Y, L,0)

    return Xapp, Yapp, Xtest, Ytest

# analyse_perfs
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne = sum(L)/len(L)
    ecart_type = np.sqrt(sum([(x-moyenne)**2 for x in L])/len(L))
    return (moyenne, ecart_type)   

# ------------------------ 

import copy

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    
    ##########################
    newC = copy.deepcopy(C)
    index = np.random.permutation(len(X)) # mélange des index
    Xm = X[index]
    Ym = Y[index]
    
    for i in range(nb_iter):
        Xapp,Yapp,Xtest,Ytest = crossval_strat(Xm, Ym, nb_iter, i)
        newC.train(Xapp, Yapp)
        perf.append(newC.accuracy(Xtest, Ytest))
        #print(i," : taille app.=  ",len(Xapp)," taille test=  ",len(Xtest)," Accuracy:  ",perf[i])
    
    ##########################
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

def leave_one_out(C, DS):
    """ Classifieur * tuple[array, array] -> float
    """
    X,Y = DS
    point = 0

    for i in range(len(X)):
        #Nouveau dataset
        x = np.delete(X,obj=i,axis=0)
        y = np.delete(Y,obj=i)
        #Apprentissage
        newC = copy.deepcopy(C)
        newC.train(x,y)
        #Accuracy
        if(newC.predict(X[i]) == Y[i]):
            point += 1
    return point/len(X)