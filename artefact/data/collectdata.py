import random
languages = ["french","german","greek","italian","japanese","latin"]

wordlists = []
for i in languages:
    wordlists.append(open(i,"r").readlines())
trainingData = open("trainData.txt","r").readlines()
for wordlist in range(len(languages)):
    print(wordlist)
    originalWordList = wordlists[wordlist]
    newWordList = []
    while len(newWordList) < 15e3:
        choice = random.choice(originalWordList)
        if choice not in newWordList and (choice.strip()+","+str(wordlist)+"\n" not in trainingData) :
            newWordList.append(choice.strip()+","+str(wordlist)+"\n")
        if len(newWordList)%100 == 0:
            print(len(newWordList))
    wordlists[wordlist] = newWordList

totalWordList = []
for i in wordlists:
    totalWordList += i

random.shuffle(totalWordList)

file = open("testData.txt","w")
for i in totalWordList:
    file.writelines(i)
file.close()
    