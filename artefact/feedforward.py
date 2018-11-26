import numpy as np
import random as r
import math as m

def letter2vec(l):
  letterPattern = [[0]]*26
  if l != "|":
    letterPattern[ord(l.lower())-97] = [1]
  return letterPattern

def word2vec(a,N):
  # returns vector with 26*N dimensions (N = 15, size = 390)
  vec = []
  if len(a) < N:
    a = a + "|"*(N-len(a))
  for i in a:
    vec += letter2vec(i)
  return np.array(vec)

class network:
  def __init__(self,hyperparameters):
    # HYPERPARAMETERS
    self.hypNames = ["N", "hiddenNodes", "hiddenLayers", "learning_rate", "languages"]
    for i in range(len(hyperparameters)):
      exec("self.{}={}".format(self.hypNames[i],str(hyperparameters[i])))
    
    # CONSTANTS
    self.layers = self.hiddenLayers + 2
    
    # setup rand w's and b's
    
    self.w = [0]*self.layers
    self.b = [0]*self.layers
    
    self.w[1] = (2*np.random.random((self.hiddenNodes,self.N*26))-1) # eg. (80,390) // (80,390).(390,1) = (80,1) 
    self.b[1] = (2*np.random.random((self.hiddenNodes,1))-1)

    for i in range(2,self.hiddenLayers   +1): # weight matrices 1-3
        self.w[i] = 2*np.random.random((self.hiddenNodes,self.hiddenNodes))-1
        self.b[i] = 2*np.random.random((self.hiddenNodes,1))-1
    self.w[-1] = 2*np.random.random((self.languages,self.hiddenNodes))-1 # eg. (1,80) // (1,80).(80,1) = (1,1) 
    self.b[-1] = 2*np.random.random((self.languages,1))-1
  
  """

  a node: -->O with layer 1, has w1, a1 and d1 where w1 is used to make a1
  """

  def train(self,word,label):
    self.word = word
    self.label = label    
    self.forprop()
    self.backprop()
  
  def prediction(self,word,numerical=False):
    if numerical == True:
      maxVal = float(max(self.a[-1]))
      return list(self.a[-1]).index(maxVal)
    
    self.word,self.label = word,0
    if len(self.word) > 15:
      self.word = self.word[:14]
    self.forprop()
    output = self.a[-1]
    temp = []
    for i in output:
      temp.append(float(i))
    output = temp
    maxVal = float( max(output) )
    guess = list(output).index(maxVal)
    languages = ["French","German","Greek","Italian","Japanese","Latin"]
    print("Guess: "+languages[guess])
    confidence = maxVal/sum(output)
    print("Confidence: {}%\n".format(round(confidence*100,1)))
    
    allGuesses = []
    
    for i in range(len(output)):
      allGuesses.append([round(output[i]*100/sum(output),1),languages[i]])
    allGuesses = sorted(allGuesses,reverse=True)
    for i in allGuesses:
      print("{}% confidence in {}".format(i[0],i[1]))
      
    
  def labelise(self,a):
    v = [[0]]*self.languages
    v[a] = [1]
    return np.array(v)
    
  def forprop(self):
    self.inp = word2vec(self.word,self.N)  # eg. (390,1)
    self.label = self.labelise(self.label)
    self.a = [0]*self.layers
    self.z = [0]*self.layers
    self.a[0] = self.inp
    self.z[0] = self.inp
    
    # a[i+1] = f(a[i])   
    # a[i+1] = w[i+1].a[i]+b[i+1]
    for i in range(self.hiddenLayers+1):
      self.z[i+1] = np.dot(self.w[i+1],self.a[i]) + self.b[i+1]
      self.a[i+1] = self.act(self.z[i+1])
     
  def backprop(self):
    d = [0]*(self.layers)
    L = -1
    self.d=d
    # BP1
    "VaC * sigma'(zL)"
    d[L] = (self.a[L]-self.label) * self.act(self.z[L],deriv=True) 
    # BP2
    for l in range(self.layers-2,0,-1):  # deltas of all layers apart from 0
      d[l] = np.dot(np.transpose(self.w[l+1]),self.d[l+1]) * self.act(self.z[l],deriv=True)
    # BP3 & BP4
    "dC/dbl = dl"
    "dc/dwl = al-1 . dl"

    for l in range(1,self.layers):
      self.w[l] -= np.dot(d[l],np.transpose(self.a[l-1]))*self.learning_rate
      self.b[l] -= d[l]*self.learning_rate
  
  def f(self,z,deriv=False,inv=False): # the activation function
    if deriv == True:
      return 1/(2*m.cosh(z) + 2)
    elif inv == True:
      return -m.log(1/z -1)
    else:
      return 1/(1+m.e**(-z))  

  def arrayf(self,M,deriv=False,inv=False):
    for y in range(len(M)):
      for x in range(len(M[0])):
        M[y][x] = self.f(M[y][x],deriv,inv)
    return M
  
  def act(self,z,deriv=False,inv=False): # elementwise activation
      if type(z) == np.ndarray:
          return self.arrayf(z,deriv,inv)
      else:
          return self.f(z,deriv,inv)