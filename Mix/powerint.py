import numpy as np
import time
import warnings

def powerint(x, p):
    """Calculate base^exponent (x^p)
    When p < 0, x^(|p|) is evaluated first and then inverted.
    This way no precision is lost due to floating point during the
    exponentiation part of the process.
    
    Arguments:
        x: Intiger or floating point number greater or equal to zero.
        p: Intiger number (different than zero iff x equals zero).
    
    Returns:
        The solution of x^p
    
    Raises:
        RuntimeError
        TypeError
        ArithmeticError
        ZeroDivisionError
        OverflowError
    """

    # Change the arguments name for easier reading through the code
    base = x
    exponent = p

    # Check that both arguments are valid.
    if base < 0:
        print(f'Your input:\n\tBase: {base}\n\tExponent: {exponent}')
        raise RuntimeError("Please provide a non negative number" \
                           " for the base.")
    elif type(base) is (not int or not float):
        print(f'Detected base number of type: {type(base)}')
        raise TypeError("Exponent must be casted to int or float to avoid" \
                        " possible over or underflows.")
    elif (type(exponent) is (not int or not np.intc or not np.int_ \
          or not np.int8 or not np.int16 or not np.int32 or not np.int64)) \
            and (type(exponent) is not type(base)):
        print(f'Detected exponent number of type: {type(exponent)}')
        raise TypeError("Exponent must be an int.")
    elif base == 0:
        if exponent == 0:
            raise ArithmeticError("Math error. Trying to perform: 0^0.")
        elif exponent < 0:
            raise ZeroDivisionError("Math error. Trying to divide by zero.")

    # Powers that are not necessary to calculate: 0^p, 1^p, x^0, x^1
    if base == 0:
        return 0
    elif base == 1 or exponent == 0:
        return 1
    elif exponent == 1:
        return base

    # Warning management
    np.seterr(all='warn')
    with warnings.catch_warnings():
        warnings.simplefilter('error')

    # Negative exponent management
    negativeExponent = False
    if exponent < 0:
        negativeExponent = True
        exponent *= -1

    result = 0

    try:
        result = _binaryExponent(base, exponent)

    except Warning:
        if negativeExponent is True:
            exponent *= -1
        raise OverflowError("Overflow while performing: ",
                            f"{base})^({exponent}).")

    if negativeExponent is True and result != 0:
        return float(1 / result)
    else:
        return result


def _binaryExponent(base, exponent):
    """Calculate base^exponent using the binary exponentiation algorithm.

    Depends on another function that verifies that all parameters are
    properly given.

    Arguments:
        base: Intiger or floating point number.
        exponent: Intiger number.

    Returns:
        base^exponent

    Raises
        RuntimeError: When trying to perform 0^0
    """
    # For an exponent <= 4 the algorithm makes no sense.
    # log_2(4) = 2 and 4 = 0b100, which will result in the exact same
    # process as the binary exponentiation algorithm.
    #
    # Avoiding it, we are actually saving a few operations (the bitwise and the
    # comparison).

    result = 1

    if (exponent == 0 and base == 0):
        raise RuntimeError("Magic error happened: 0^0.")
    elif (exponent == 0 and base != 0):
        return 1
    elif (exponent < 5):
        for _ in range(exponent):
            result *= base
    else:
        while (exponent > 0):
            # Current bit = 1
            if (exponent & 1):
                result *= base
            base *= base
            exponent = exponent >> 1

    return result


def test_powerint():
    # Valid combinations of x^p powerint

    # For powerint
    simpleIntPowers = (
        ([0, 1]), ([0, 5]), ([0, 17]), ([0, 752]),
        ([2, 0]), ([1, 0]), ([1435, 0]), ([523, 0]),
        ([1, 1]), ([1, 5]), ([1, -5223]), ([1, -3]),
        ([2, 2]), ([2, 15]), ([2, 5]),
        ([2, -5]), ([2, -58]),
        # Some random-generated sets (using random.org)
        ([51, -10]), ([5403, -3]),
        ([43, 36]), ([95, 8]),
        ([27, 10]), ([32, 5]),
        ([76, -12]), ([67, 10]),
        ([31, 9]), ([57, -9]),
        ([17, 14]), ([38, -60]),
        ([52, 7]), ([10, -8]),
        ([28, -25]), ([69, 4]),
        ([67, 10]), ([75, 3]),
        ([93, 8]), ([41, -2]),
        ([24, -4]), ([37, 9]),
        ([82, -38]), ([48, -5]),
        ([18, -17]), ([50, -19]),
        ([2, 20]), ([7, -3]),
        ([90, 20]), ([592, 94]),
        ([819645, -369723]), ([22962, 396793]),
        ([423837, -78785]), ([562804, 447158]),
        ([506033, 1233]), ([864805, -7011]),
        ([382783, -713864]), ([873793, -974258]),
        ([381540, -262639]), ([152469, -173293]),
    )

    simpleFloatPowers = (
        ([0.0, 1]), ([0.0, 17]),
        ([0.0, 752]),
        ([2.0, 0]), ([1.0, 0]),
        ([1435.0, 0]),
        ([523.0, 0]),
        ([1.0, 1]), ([1.0, 5]),
        ([1.0, -5223]),
        ([1.0, -3]),
        ([2.0, 2]), ([2.0, 15]),
        ([2.0, -5]), ([2.0, -58]),
        # Some random-generated sets (using random.org)
        ([22652.6, 30]), ([30.55, -37]),
        ([902.599, -2]), ([385.349, 47]),
        ([70.846, 91]), ([28.341, 21]),
        ([51.886, -16]), ([543.81, -69]),
        ([43.19, -36]), ([95.331, 8]),
        ([271.991, 10]), ([3220.9, -25]),
        ([76.57, -12]), ([63.973, 12]),
        ([31.928, 9]), ([59.695, -9]),
        ([137.350, 14]), ([338.48, -60]),
        ([542.261, 17]), ([10.866, -8]),
        ([28.265, -25]), ([569.852, 4]),
        ([67.249, 36]), ([75.968, 3]),
        ([923.4, 81]), ([414.84, 2]),
        ([24.15, -4]), ([370.314, 9]),
        ([82.880, -38]), ([48.97, -5]),
        ([188.91, -17]), ([50.177, -19]),
        ([2.360, 20]), ([7.376, -3]),
        ([90.424, 20]), ([59.912, -9]),
    )

    numberString = ''

    # Tests for powerint
    print('=' * 60, "\nTesting powerint()...\n", '-' * 59)
    # Invalid combinations of x^p

    # Valid integer combinations of x^p
    for base, exponent in simpleIntPowers:
        numberString = f'({base})^({exponent})'
        print("Testing: {0:<30}\t\t".format(numberString), end=' ')
        test_powerint_assert(base, exponent)
        print("-> Pass <-")

    # Valid floating point combinations of x^p (p is always integer)
    for base, exponent in simpleFloatPowers:
        numberString = f'({base})^({int(exponent)})'
        print("Testing: {0:<30}\t\t".format(numberString), end=' ')
        test_powerint_assert(base, exponent, dataType=float)
        print("-> Pass <-")


def test_powerint_assert(base, exponent, dataType=int):
    """Test case for powerint function.
    Calls powerint and compares its answer with python's exponentiation
    function.
    Whenever the exponent is a negative number, numpy's allclose function
    is used instead to deal with floating point numbers.
    This function will stop the code execution when it finds an error.
    Returns:
        None
    Raises:
        Nothing
    """

    ans = powerint(base, exponent)
    # print(ans)

    # This is not meant to be fast, but to be simple and not failing.
    if base == 0:
        pyPow = 0
    elif exponent < 0 and base != 0:
        pyPow = 1 / (base ** ((-1) * exponent))
    else:
        pyPow = base ** exponent

    # For debuging purposes only
    # print('{0} || {1}'.format(ans, pyPow))

    # assert True, "X ERROR X"
    if dataType == int and exponent > 0:
        assert ans == pyPow, "X ERROR X"
    else:
        assert np.allclose(ans, pyPow), "X ERROR X"


if __name__ == "__main__":
    print("Running tests...")
    initialTime = time.time_ns()

    test_powerint()

    finalTime = time.time_ns()

    print("\nTests ran in: %f s (%d ns)\n" % ((finalTime - initialTime) \
                                            * 10 ** (-9),
                                            finalTime - initialTime))