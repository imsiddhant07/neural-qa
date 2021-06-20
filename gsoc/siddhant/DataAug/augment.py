import syntax_aware
import utility

#Reading Data from text file
file_en = "data-en.txt"
file_sp = "data-sp.txt"
text_en = utility.return_data(file_en)
text_sp = utility.return_data(file_sp)


#Sampling a subset
setsize = 10
shuffled_en, shuffled_sp = utility.randomset(text_en, text_sp, setsize)

en_sp = []

for i in range(len(shuffled_en)):
    en_sp.append(syntax_aware.getData(shuffled_en[i], shuffled_sp[i]))


en_prob = []   

for i in range(len(shuffled_en)):
    en_prob.append(syntax_aware.parsingTree(en_sp[i], 0.1))



