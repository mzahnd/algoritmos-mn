#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def _expSum(x, iterations=1000):
    """Calculate e^x using power series.

    exp x := Sum_{k = 0}^{inf} x^k/k! = 1 + x + x^2/2! + x^3/3! + x^4/4! + ...
 
    Which can be rewritten as:
                                    = 1 + ((x/1)(1 + (x/2)(1 + (x/4)(...) ) ) )

    This second way of writting it is easier to calculate than the first one
    as it does not need to directly calculate the factorial of each term.

    Arguments:
        iterations: Times to iterate over the exponential power series.
                    The minimum valid number is 1 (one), and it'll return the
                    equivalent to perform 1 + x.
                    Default is 1000 as it does not take much time to complete
                    (even for big numbers such as e^500) and going beyond that
                    does not make a significative difference.
                    e^500, e^2, e^50, and some  other tried examples get the 
                    same number up to 14 decimal places using 1000 (the 
                    default) and  1000000 (the default value squared) 
                    iterations.

    Returns:
        Floating point number.

    Raises:
        ArithmeticError: When trying to iterate less than one time.

    """
    
    if type(x) is (not int or not float):
        raise ArithmeticError('Please provide an int or float.')
    
    if (iterations < 1):
        raise ArithmeticError('At least one iteration needed to calculate e^x')

    # e^0 = 1
    if (x == 0):
        return float(1.0)

    isNegative = False

    # The algorithm always calculates e^x (x > 0) and then divides 1 by the
    # result if x < 0. This avoids missing extra precission due to floating 
    # point.
    if (x < 0):
        isNegative = True
        x *= -1

    result = float(1.0)

    for num in range(iterations, 0, -1):
        # In the power series: = 1 + ((x/1)(1 + (x/2) (1 + (x/4) (...) ) ) )
        # x/num is the same as (x/4), or (x/2), (x/1); result is the rightmost
        # part of the series, which has been already calculated.
        result = 1 + ((x * result)/num)

    if isNegative:
        result = float(1/result)

    return float(result)

if __name__ == "__main__":
    ans = _expSum(2.5, iterations=1000)
    print(ans)