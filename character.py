from collections import Counter 
from itertools import chain 
from PIL import Image 
import numpy as np 
from hmm import viterbi_inference
def read_text(filename):
  with open(filename, 'r', encoding='utf8') as f:
    return f.read()

def bigrams(line):
  return zip(line, line[1:])

def bigram_counts(texts):
  texts = texts + [text.lower() for text in texts] + [text.upper() for text in texts]
  return Counter(chain.from_iterable(bigrams(text) for text in texts))

def char_counts(texts):
  return Counter(chain.from_iterable(texts))

def initial_char_counts(texts):
  return Counter(text[0] for text in texts if len(text) > 4 )




CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
def read_images(filename):
  img = Image.open(filename)
  x_max, y_max = img.size

  pixels = np.array(img)
  width = int(x_max / CHARACTER_WIDTH) * CHARACTER_WIDTH
  
  return [pixels[:, x:x+CHARACTER_WIDTH] 
        for x in range(0, width, CHARACTER_WIDTH)]



from os import path
import matplotlib.pyplot as plt

char_images = read_images(path.join('data', 'courier-train.png'))
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

texts = read_text(path.join('data', 'english.txt')).splitlines()

cc = char_counts(texts)
bc = bigram_counts(texts)

c0 = initial_char_counts(texts)
c0_total = sum(c0.values())
alpha = 1

def log_p0(c):
  ''' computes the log of initial state probability '''
  # p(c) = (#c + alpha)/(total + alpha*(# of possible c))
  c = chars[c]
  return np.log(c0.get(c, 0) + alpha) - np.log(c0_total + alpha*len(chars))

def log_t(previous, current):
  ''' computes the log transition probability
    Character-transition model
    Assumption: Each character t+1 is conditionally independent of all other characters given the character t
  '''
  # p(current|previous) = (#(previous,current) + alpha)/(#(previous) + alpha*(# of possible current) )
  previous, current = chars[previous], chars[current]
  joint = bc.get((previous, current), 0)
  
  marginal = cc.get(previous, 0)
  return np.log(joint + alpha) - np.log(marginal + alpha*len(chars))

def log_e(char, image):
  ''' computes the log-probability that a specific image was observed for a given character 

    Image-Emission model summary:
    Assumption: Each pixel is conditionally independent given the observed character
      Likelihood: pixel-matched ~ Binomial(n, mu)
      Prior: mu ~ Beta(alpha, beta)
      Posterior: mu|data ~ Beta(alpha+x, beta+n-x)
      Bayesian Solution: P(pixel-matched|mu) =  (alpha + x) / (alpha + beta + n)
      where x: number of pixels that matched
            n: total number of pixels
        alpha: pseudocount for matched pixels
        beta: pseudocount for pixels that were not matched
      If, we have seen no data at all
        P(pixel-matched|mu) =  (alpha) / (alpha + beta)
  '''

  # punctuation can be confused with salt-and-pepper noise
  mu = 0.65 if chars[char] in ' ,.\'"-' else 0.7 

  
  char_image = char_images[char]
  
  height, width = char_image.shape
  n = height * width

  original_pixels = chain.from_iterable(char_image[1:-1])
  observed_pixels = chain.from_iterable(image[1:-1])
  count = sum(
    original == observed
    for original, observed in zip(original_pixels, observed_pixels)
  )
  
  return count*np.log(mu) + (n-count)*np.log(1-mu)

for i in range(0, 20):
  test_images = read_images(path.join('data', 'test-%d-0.png' % i))
  sequence = viterbi_inference(test_images, list(range(len(chars))), log_p0, log_e, log_t)

  print (''.join(chars[i] for i in sequence))


'''

m = np.zeros((len(chars), len(chars)))
for i, ci in enumerate(chars):
  for j in range(i+1):
    count = sum(
      original == observed
      for original, observed in zip(chain.from_iterable(char_images[i][1:-1]), chain.from_iterable(char_images[j][1:-1]))
    )
    m[i,j] = m[j,i] = count
'''