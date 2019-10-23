import numpy as np

def viterbi_inference(data, states, log_p0, log_e, log_t):
  ''' performs MAP inference on a hidden markov model '''
  viterbi = np.zeros((len(data), len(states)), dtype=float)
  backtrack = np.zeros((len(data), len(states)), dtype=int)

  viterbi[0] = [log_p0(state) + log_e(state, data[0]) for state in states]
  for i, obs in enumerate(data[1:], start=1):
    for j, state in enumerate(states):
      options = [viterbi[i-1,p] + log_t(previous, state) for p, previous in enumerate(states)]
      max_i = np.argmax(options)
      backtrack[i,j] = max_i
      viterbi[i,j] = options[max_i] + log_e(state, obs)
  current = np.argmax(viterbi[-1])
  result = [states[current]]
  for row in backtrack[:0:-1]: # all rows except 0th row, in reverse order
      current = row[current]
      result.append(states[current])
  return result[::-1]

