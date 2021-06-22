import numpy as np
from numpy.core.fromnumeric import shape
import syntax_aware
import utility

#Reading Data from text file
file_en = "/home/siddhant/neural-qa/gsoc/siddhant/DataAug/data-en.txt"
file_sp = "/home/siddhant/neural-qa/gsoc/siddhant/DataAug/data-sp.txt"
text_en = utility.return_data(file_en)
text_sp = utility.return_data(file_sp)


#Sampling a subset
setsize = 2
shuffled_en, shuffled_sp = utility.randomset(text_en, text_sp, setsize)

en_sp = []
for i in range(len(shuffled_en)):
    en_sp.append(syntax_aware.getData(shuffled_en[i], shuffled_sp[i]))


en_prob = []   
for i in range(len(shuffled_en)):
    en_prob.append(syntax_aware.parsingTree(en_sp[i], 0.1))


replaced_data = []
for i in range(len(en_prob)):
    inde1, inde2 = syntax_aware.top2(en_prob[i])
    replaced_data.append(syntax_aware.dropout(en_sp[i][0], inde1, inde2))

en_sp_replaced = []
for i in range(len(replaced_data)):
    en_sp_replaced.append(syntax_aware.getData(replaced_data[i], shuffled_sp[i]))

print(en_sp)
print(en_sp_replaced)

f = open("/home/siddhant/neural-qa/gsoc/siddhant/DataAug/extended_data-en.txt", "w+")
for i in range(len(en_sp_replaced)):
    f.write(en_sp_replaced[i][0]+"\n")

f.close()
