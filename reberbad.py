"""Helper functions that generate a single string looking
   identical to, but not following Embedded Reber Grammar"""


import random

# Returns the non-embedded, internal string
def reber_prime_bad():

    string = "B"
    luck = random.random()

    if luck < 0.5:
        string = reber_phase_a_bad(string)
        luck = random.random()
        if luck < 0.5:
            string += "T"
            string = reber_phase_b_bad(string)
        else:
            string += "SE"

    else:
        string += "P"
        string = reber_phase_b_bad(string)

    return string

# Returns a portion of the string corresponding to the top half of the graph
def reber_phase_a_bad(string):
    string += "X"
    luck = random.random()

    while luck < 0.5:
        string += "S"
        luck = random.random()

    string += "P"

    return string

# Returns a portion of the string corresponding to the bottom half of the graph
def reber_phase_b_bad(string):
    string += "T"
    luck = random.random()

    while luck < 0.5:
        string += "T"
        luck = random.random()

    string += "V"

    luck = random.random()
    if luck < 0.5:
        string += "PE"
        luck = random.random()
        if luck < 0.5:
            string += "X"
            string = reber_phase_b_bad(string)
        else:
            string += "SE"

    else:
        string += "VE"

    return string

# Returns a complete, false embedded reber grammar string
def embedded_reber_bad():
    string = "B"
    luck = random.random()

    if luck < 0.5:
        string += "P"
        string += reber_prime_bad()
        luck = random.random()
        if luck < 0.5:
            string += "PE"
        else:
            string += "TE"

    else:
        string += "T"
        string += reber_prime_bad()
        if luck < 0.5:
            string += "TE"
        else:
            string += "PE"

    return string