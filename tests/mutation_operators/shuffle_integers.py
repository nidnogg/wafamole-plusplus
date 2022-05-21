import re
import random

def shuffle_integers(payload):
  candidates = list(re.finditer(r'[0-9]', payload))

  if not candidates:
    return payload
  candidate_pos = random.choice(candidates).span()

  return payload[: candidate_pos[0]] + str(random.choice(range(10))) + payload[candidate_pos[1] :]


def mutation_round():  
  payload = "1234admin' or 1=1#"
  i = 0
  print("starting with {}".format(payload))
  while i < 20:
    payload = shuffle_integers(payload)
    print(payload)  
    i += 1

mutation_round()