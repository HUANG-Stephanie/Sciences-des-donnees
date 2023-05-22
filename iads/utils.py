# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ------------------------ 

# genere_dataset_uniform:
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    valeur = np.random.uniform(binf,bsup,(n*2,p))
    label = np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,n)])
    return (valeur, label)

# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    negative = np.random.multivariate_normal(negative_center, negative_sigma, size = nb_points)
    positive = np.random.multivariate_normal(positive_center, positive_sigma, size = nb_points)
    data_desc = np.concatenate((negative, positive))
    data_labels = np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
    return (data_desc, data_labels)

# genere_train_test
def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """
    li_1 = np.where(label_set == 1)
    li_2 = np.where(label_set == -1)
    
    L1 = random.sample([i for i in li_1[0]],n_pos)
    L2 = random.sample([i for i in li_2[0]],n_neg)
    L = L1 + L2
  
    #base d'apprentissage
    Xtrain = desc_set[L]
    Ytrain = label_set[L]
    #base de test
    Xtest = np.delete(desc_set,L,0)
    Ytest = np.delete(label_set,L,0)
    
    return (Xtrain, Ytrain), (Xtest, Ytest)

# plot2DSet:
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
   #TODO:
    data_negatifs = desc[labels == -1]
    data_positifs = desc[labels == +1]
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="cornflowerblue") 
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="pink") 

# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

# create_XOR 
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    data_xor1 = np.random.multivariate_normal([0,0], [[var,0],[0,var]], n)
    data_xor2 = np.random.multivariate_normal([1,1], [[var,0],[0,var]], n)
    data_xor3 = np.random.multivariate_normal([1,0], [[var,0],[0,var]], n)
    data_xor4 = np.random.multivariate_normal([0,1], [[var,0],[0,var]], n)
    label_xor1 = np.array([-1]*n)
    label_xor2 = np.array([-1]*n)
    label_xor3 = np.ones(n)
    label_xor4 = np.ones(n)
    data_xor = np.concatenate((data_xor1, data_xor2, data_xor3, data_xor4))
    label_xor = np.concatenate((label_xor1, label_xor2, label_xor3, label_xor4))
    return data_xor, label_xor

