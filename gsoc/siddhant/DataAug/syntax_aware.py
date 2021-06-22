"""
Part of dbpedia/neural-qa 
Code for implementing Syntax Aware Data Augmentation on DBNQA 

"""

from networkx.classes import ordered
from numpy.core.records import array
from numpy.lib.function_base import copy
import spacy
import networkx as nx
import math
import numpy as np

nlp = spacy.load("en_core_web_sm")


def getData(textEn, textSP):
    pair = []
    pair.append(textEn)
    pair.append(textSP)
    return pair


def parsingTree(pair, alpha):
    doc = nlp(pair[0])
    edges = []
    nodes = []
    root_word = ""
    for w in doc:
        edges.append((w.head.text, w.text))
        nodes.append(w.text)
        if w.dep_ == "ROOT":
            root_word = w.text
    
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    
    targetWords = []
    wordDepths = []

    for node in list(graph.nodes):
        #target = node
        try:
            d = nx.shortest_path_length(graph, source=root_word, target=node)
            targetWords.append(node)
            wordDepths.append(d)
        except nx.exception.NetworkXNoPath:
            pass
#        targetWords.append(node)
#        wordDepths.append(d)

    prob = []

    for wD in wordDepths:
        if wD == 0:
            pass
        else:
            power = wD - 1
            den = math.pow(2, power)
            frac = 1/den
            probab = 1 - frac
            prob.append(probab)

    sigm = []
    prob_ar = np.array(prob)
    for prob_a in prob_ar:
        sig = 1/(1+math.exp(-(prob_a)))
        sigm.append(sig)
    
    sigmoid = np.array(sigm)
    
    probFinal = sigmoid * alpha * float(len(sigm))




    return probFinal


def top2(probs):
    arr = probs
    index1 = np.argmax(arr)
    arr[index1] = 0.0
    index2 = np.argmax(arr)
    return index1, index2


def dropout(en_sen, inde1, inde2):
    en_words = en_sen.split()
    en_words[inde1] = ""
    en_words[inde2] = ""
    final = " ".join(en_words)

    return final


def replacement():
    pass



def blanking():
    pass






if "__name__" == "__main__":
    pass

