import os
import time
import datetime
import csv
from feedforward import *

def fileNum(a,_type_): # strips back file names to get the weight or bias number 
                       # (for importing w's and b's)
  if _type_ == "weight":
    return int(a[6:len(a)-4])
  elif _type_ == "bias":
    return int(a[4:len(a)-4])
    
def load(cache,fromCache=True): # loads weights and biases from a file stored network (cache)
  # setting path to retrieve weights and biases from
  if fromCache == True:
    cachePath = os.path.join("cache",cache)
  else:
    cachePath = cache
    
  weightPath = os.path.join(cachePath,"weights")
  biasPath = os.path.join(cachePath,"biases")  
  weightNames, biasNames = os.listdir(weightPath), os.listdir(biasPath)
  
  # loading weights
  weightNums = []
  for fileName in weightNames:
    weightNums.append( int(fileName[6:-4]) )
  weights = [0]*( len(weightNames) +1 )
  for i in weightNums:
    weights[i] = np.load(os.path.join(weightPath,weightNames[i-1]))['w']
  h.w = weights
  # loading biases
  biasNums = []
  for fileName in biasNames: # ie. "13.npz" (str) ==> 13 (int) 
    biasNums.append(int(fileName[4:-4]))
  biases = [0]*( len(biasNames) +1 )
  for i in weightNums:
    biases[i] = np.load(os.path.join(biasPath, biasNames[i-1]))['b']
  h.b = biases

def findNewCache(): # for a programmer to save network to a file
                    # makes a new folder to save progress of current program, contains weights and biases folders
  allCaches = []
  for i in os.listdir("cache"):
    allCaches.append(int(i))
  if len(allCaches) > 0:
    cacheName = str(max(allCaches)+1)
  else:
    cacheName = "0"
  path = os.path.join("cache",cacheName)
  if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(os.path.join(path, "weights"))
    os.makedirs(os.path.join(path,"biases"))
  return cacheName

def saveProgress(done=False):
  global cacheName,word_index,loop,trainingLoops,correct,incorrect
  
  # Save weights and biases
  for i in range(1,len(h.w)):
    np.savez(os.path.join("cache",cacheName,"weights",str(i)),w=h.w[i])

  for i in range(1,len(h.b)):
    np.savez(os.path.join("cache",cacheName,"biases",str(i)),b=h.b[i])
  
  # Save current position
  with open(os.path.join("cache",cacheName,"progress.txt"),"w") as file:
    if done == True:
      file.writelines("Completed running of {} loops\n".format(trainingLoops))
    else:
      file.writelines("Finished loop {} out of {}".format(loop+1,trainingLoops))
      
  # Save performance of network at current iteration
  with open(os.path.join("cache", cacheName, "performance.csv"),"a",newline='') as file:
    accuracy = correct/(correct+incorrect)
    csv.writer(file).writerow([accuracy,loop+1])

def saveHyps(): # saving hyperparameters of network to a file
  global cacheName,hyperparameters
  with open(os.path.join("cache", cacheName, "hyperparameters.txt"),"w") as file:
    file.writelines(str(hyperparameters)+"\n")
    file.writelines(str(h.hypNames)+"\n")  
  
def reformat(a): # each line in a file is written as eg. "ransack,3" 
  commPos = a.index(",")
  word = a[:commPos]
  num = int(a[commPos+1:])
  return [word,num] # outputs eg. "ransack,3" (str) ==> ["ransack",3] (list)

def ETA(): # returns at what 24 hour time the program expects to complete at
  global start,loop,trainingLoops,word_index
  now = time.time()
  timeElapsed = now - start # secs
  wordsElapsed = word_index + loop*120e3
  wordsLeft = trainingLoops*120e3 - wordsElapsed
  timeLeft = (timeElapsed/wordsElapsed)*wordsLeft
  timeCurrent = now%(24*60*60)
  timeFinished = int(timeCurrent + timeLeft) 
  timeFinished -= timeFinished%60 # removes seconds ie. ETA always AB:CD:00 so no extra accuracy is suggested
  return str(datetime.timedelta(seconds = timeFinished))

def getData(name): # reads train/test file and reformats each line into a usable data pair 
  data = open(name,"r").readlines()
  for i in range(len(data)):
    data[i] = reformat(data[i])
  return data

# initialise network
data = getData(os.path.join("data","trainData.txt"))
testData = getData(os.path.join("data","testData.txt"))

hyperparameters = [15,200,1,0.001,6]
h = network(hyperparameters)
trainingLoops = 20

# can load a network with the same shape by referencing cache it is stored in - ie. load("9") or load("WandBsaves\\200n1h",fromCache=False)
load(os.path.join("data", "WandBsaves","200n1h"),fromCache=False)

while True:
  print("\nInput a word and I'll guess what langauge it is.\nThe language must be French, Greek, German, Italian, Latin or Japanese\n")
  word = input(">>> ")
  print()
  h.prediction(word)
  print("-"*30)


# Putting speech marks """ around  the load(..) line and the while loop above, the programmer can remove the speech marks around the following section. This will let him/her train and test a network and save necessary parts to a file after each iteration. 
# This is what I used to train and test my network and output the results.
"""
cacheName = findNewCache()
saveHyps()
### train network with a certain shape and learning rate, this would output all recorded data to a cache store
print("Training...")
start = time.time() # for ETA
print("Doing network with {} neuron(s) and {} hidden layer(s)".format(hyperparameters[1],hyperparameters[2]))

for loop in range(trainingLoops):
  # Training network on train set
  for word_index in range(len(data)):
    wordLabelPair = data[word_index]
    if len(wordLabelPair[0]) <= h.N:
      h.train(wordLabelPair[0],wordLabelPair[1])
  print("Training Completed.\nTesting...")
  
  # Testing network on test set
  correct,incorrect = 0,0
  for testItem in testData:
    h.word,h.label = testItem[0],testItem[1]
    label = h.label
    if len(h.word) <= h.N:
      h.forprop()
      if label == h.prediction("",numerical=True):
        correct += 1
      else:
        incorrect += 1
        
  # Formatting printing for terminal
  finished = (loop == trainingLoops-1)
  if not finished:
    appendText = "\nTraining..."
    done = False
  else:
    appendText = ""
    done = True
    print("Training and testing complete.")
    continue
  print("Testing Completed.\nETA " + ETA()+"\nCompleted loop "+str(loop+1)+"\n-------------------"+appendText)
  
  # save weights, biases, test set accuracy, etc.
  saveProgress(done)
"""
