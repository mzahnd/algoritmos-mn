#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def binf2dec(binArr):
    if (type(binArr) is not np.ndarray ):
        raise RuntimeError("Invalid type.")
    
    isNegative = False
    finalNumber = 0

    # 16 bits
    # 16 bit: bias = 15 ( 2^(5-1) - 1 = 15)
    expSize = 5
    mantSize = 10
    bias = 2**(expSize-1)-1

    # Hardcoded maximum exponent number (all 1)
    # For 16 bits (with expSize = 5), this is equal to
    # 2^4 + 2^3 + 2^2 + 2^1 + 2^0
    maxExponentPower = 31

    if (binArr[0] == 1):
        isNegative = True

    # Calculate exponent
    exponent = 0
    
    exponentPowers = 0
    for pot in range(1, expSize + 1, 1):
        if (binArr[pot]):
            exponentPowers += 2**(expSize - pot)

    # Calculate mantissa
    mantissa = 0

    for mant in range(expSize + 1, mantSize + expSize + 1, 1):
        if (binArr[mant]):
            # - (mant + expSize) = expSize - mant
            # For 16 bits: - (mant + 5 ) = 5 - mant
            mantissa += 2**(expSize - mant)

    if (exponentPowers == 0 and mantissa == 0):
        # Zero
        finalNumber = 0

    elif (exponentPowers == 0 and mantissa != 0):
        #Sub-Normal number
        exponent = 2**(1 - bias)
        finalNumber = exponent * (0 + mantissa)
        if (isNegative):
            finalNumber *= -1

    elif (exponentPowers > 0 and exponentPowers < maxExponentPower):
        # Normal number
        exponent = 2**(exponentPowers - bias)
        finalNumber = exponent * (1 + mantissa)
        if (isNegative):
            finalNumber *= -1

    elif (exponentPowers == maxExponentPower and mantissa == 0):
        # Infinity
        if (isNegative):
            finalNumber = "-inf"
        else:
            finalNumber = "inf"
    else:
        # NaN
        finalNumber = "NaN"

    return finalNumber

def test():
    testValues = {
        # Zero
        0: np.zeros((16,), dtype=int),
        # + int
        1022: np.array([0,
                        1, 1, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
        1460: np.array([0,
                        1, 1, 0, 0, 1,
                        0, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
        21248: np.array([0,
                         1, 1, 1, 0, 1,
                         0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
        48736: np.array([0,
                         1, 1, 1, 1, 0,
                         0, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
        65504: np.array([0,
                         1, 1, 1, 1, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # - int
        -1022: np.array([1,
                         1, 1, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
        -1460: np.array([1,
                         1, 1, 0, 0, 1,
                         0, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
        -21248: np.array([1,
                          1, 1, 1, 0, 1,
                          0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
        -48736: np.array([1,
                          1, 1, 1, 1, 0,
                          0, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
        -65504: np.array([1,
                          1, 1, 1, 1, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # + float
        0.0078125: np.array([0,
                             0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        0.015625: np.array([0,
                            0, 1, 0, 0, 1,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        1.129225162: np.array([0,
                               0, 1, 1, 1, 1,
                               0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
        65.551895721: np.array([0,
                                1, 0, 1, 0, 1,
                                0, 0, 0, 0, 0, 1, 1, 0, 0, 1]),
        7.401816386: np.array([0,
                               1, 0, 0, 0, 1,
                               1, 1, 0, 1, 1, 0, 0, 1, 1, 1]),
        # - float
        -0.0078125: np.array([1,
                              0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        -0.015625: np.array([1,
                             0, 1, 0, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        -1.129225162: np.array([1,
                                0, 1, 1, 1, 1,
                                0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
        -65.551895721: np.array([1,
                                 1, 0, 1, 0, 1,
                                 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]),
        -7.401816386: np.array([1,
                                1, 0, 0, 0, 1,
                                1, 1, 0, 1, 1, 0, 0, 1, 1, 1]),
        # + Subnormal numbers
        0.0000094940765: np.array([0,
                                   0, 0, 0, 0, 0,
                                   0, 0, 1, 0, 0, 1, 1, 1, 1, 1]),
        0.00000804662704: np.array([0,
                                    0, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 1, 1, 1]),
        0.0000520648489: np.array([0,
                                   0, 0, 0, 0, 0,
                                   1, 1, 0, 1, 1, 0, 1, 0, 0, 1]),
        # - Subnormal numbers
        -0.0000094940765: np.array([1,
                                    0, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 1, 1, 1, 1, 1]),
        -0.00000804662704: np.array([1,
                                     0, 0, 0, 0, 0,
                                     0, 0, 1, 0, 0, 0, 0, 1, 1, 1]),
        -0.0000520648489: np.array([1,
                                    0, 0, 0, 0, 0,
                                    1, 1, 0, 1, 1, 0, 1, 0, 0, 1]),
        # Inf
        "inf": np.append([0,
                          1, 1, 1, 1, 1],
                          np.zeros((10,), dtype=int)),
        "-inf": np.append([1,
                          1, 1, 1, 1, 1],
                          np.zeros((10,), dtype=int)),
        # NaN
        "nan": np.append([0,
                          1, 1, 1, 1, 1,
                          1], np.zeros((9,), dtype=int))
    }

    print("Performing tests...")

    for testNumber in testValues:
        print(f'Testing testNumber: {testNumber}')
        print(f'With binary representation: {testValues[testNumber]}')

        recivedNumber = binf2dec(testValues[testNumber])

        print(f"Recived: {recivedNumber}")
        if (type(recivedNumber) is str):
            comparison = (recivedNumber.lower() == testNumber.lower())
            
        else:
            comparison = np.isclose(recivedNumber, testNumber, rtol=1e-3)
        
        assert comparison, "The last tested value did not pass the test."
        print("--> Pass <--\n")

if __name__ == "__main__":
    print("Executed as stand-alone script. Running test function.\n")
    test()
