import random


def return_data(file):
    texts = []
    with open(file, "rt") as myfile:
        for myline in myfile:
            myline = myline.strip()
            texts.append(myline)

    return texts


def randomset(text_en, text_sp, setsize):
    shuffle_en = []
    shuffle_sp = []
    index = random.sample(range(1, len(text_en)), setsize)
    for ind in index:
        shuffle_en.append(text_en[ind])
        shuffle_sp.append(text_sp[ind])

    return shuffle_en, shuffle_sp
