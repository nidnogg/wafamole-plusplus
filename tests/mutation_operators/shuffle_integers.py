import re
import random


def shuffle_bases(payload):
    candidates = list(re.finditer(r'[0-9]+', payload))

    if not candidates:
        return payload
    candidate_pos = random.choice(candidates).span()
    candidate = payload[candidate_pos[0]:candidate_pos[1]]

    replacements = [
        bin(int(candidate)),
        int(candidate),
        oct(int(candidate)),
        hex(int(candidate)),
    ]

    replacement = random.choice(replacements)

    if (str(candidate) == str(replacement)):
        return payload
    return payload[:candidate_pos[0]] + str(
        replacement)[2:] + payload[candidate_pos[1]:]


def mutation_round():
    payload = "1234admin' or 1=1#"
    i = 0
    print("starting with {}".format(payload))
    while i < 20:
        payload = shuffle_bases(payload)
        print(payload)
        i += 1

mutation_round()