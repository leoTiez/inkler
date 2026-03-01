from inkler.constants import ROMAN_MAP


def roman_to_int(s: str):
    res = 0
    i = 0
    while i < len(s):
        if s[i] not in ROMAN_MAP:
            return None
        if i + 1 < len(s) and s[i + 1] not in ROMAN_MAP:
            return None

        # if the current value is less than the next value,
        # subtract current from next and add to res
        if i + 1 < len(s) and ROMAN_MAP[s[i]] < ROMAN_MAP[s[i + 1]]:
            res += ROMAN_MAP[s[i + 1]] - ROMAN_MAP[s[i]]

            # skip the next symbol
            i += 1
        else:

            # otherwise, add the current value to res
            res += ROMAN_MAP[s[i]]
        i += 1

    return res
