"""Helper functions that generate a single string
   following Embedded Reber Grammar"""

import random

# Returns the non-embedded, internal string
def reber_prime():

    string = "B"
    luck = random.random()

    if luck < 0.5:
        string = reber_phase_a(string)
        luck = random.random()
        if luck < 0.5:
            string += "X"
            string = reber_phase_b(string)
        else:
            string += "SE"

    else:
        string += "P"
        string = reber_phase_b(string)

    return string


# Returns a portion of the string corresponding to the top half of the graph
def reber_phase_a(string):
    string += "T"
    luck = random.random()

    while luck < 0.5:
        string += "S"
        luck = random.random()

    string += "X"

    return string


# Returns a portion of the string corresponding to the bottom half of the graph
def reber_phase_b(string):
    string += "T"
    luck = random.random()

    while luck < 0.5:
        string += "T"
        luck = random.random()

    string += "V"

    luck = random.random()
    if luck < 0.5:
        string += "P"
        luck = random.random()
        if luck < 0.5:
            string += "X"
            string = reber_phase_b(string)
        else:
            string += "SE"

    else:
        string += "VE"

    return string

# Returns a complete, embedded reber grammar string
def embedded_reber():
    string = "B"
    luck = random.random()

    if luck < 0.5:
        string += "P"
        string += reber_prime()
        string += "PE"

    else:
        string += "T"
        string += reber_prime()
        string += "TE"

    return string
