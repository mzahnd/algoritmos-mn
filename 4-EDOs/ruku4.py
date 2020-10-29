#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sources:
    Binary exponentiation algorithm:
    https://cp-algorithms.com/algebra/binary-exp.html
    https://eli.thegreenplace.net/2009/03/21/efficient-integer-exponentiation-algorithms
"""
import math
import numpy as np
from scipy.integrate import solve_ivp  # Used for testing
import time


def ruku4(f, x0, t0, tf, h):
    """Custom implementation of the Runge-Kuttta 4 algorithm.
    
    This function can be used as a normal RuKu 4 implementations, but it can
    also make the whole 't' and 'x' arrays visible to other functions through
    the following arrays:
        _ruku4_tArr
        _ruku4_x
    This allows your problem's function to access and read any needed data from
    the ongoing process.
    
    To enable this function, uncomment the corresponding lines.
    As it is provided, there are no global variables
    
    It is not recommended modify the values inside any of the global arrays, as
    it will leave to invalid outputs.
    
    Arguments:
        f: Function handler. Must recieve two arguments, t and x. f(t, x).
        x0: Initial value
        t0 / tf: Initial and final time to evaluate
        h: Desiered step

    Returns:
        t, x
        t: Array with the time points used for the function's calculation. 
            (X axis on a plot)
            Shape: (n,)
        x: Obtained values at a given t.
            (Y axis on a plot)
            Shape: (n,1)  - One row for each element in t -
    """

    # Uncomment the following to make it global
    # global _ruku4_tArr
    # global _ruku4_x

    # Let's make this function a little bit more readable
    step = h
    
    nPoints = int((tf-t0)/h)
    _ruku4_tArr = np.linspace(t0, tf, nPoints + 1)
    _ruku4_x = np.zeros((nPoints + 1, x0.shape[0]))

    _ruku4_x[0,:] = x0

    stepOver2 = step / 2
    for k in range(0, nPoints, 1):
        f1 = f(_ruku4_tArr[k],
                _ruku4_x[k,:])

        f2 = f(_ruku4_tArr[k] + stepOver2,
                _ruku4_x[k,:] + stepOver2 * f1)

        f3 = f(_ruku4_tArr[k] + stepOver2,
                _ruku4_x[k,:] + stepOver2 * f2)

        f4 = f(_ruku4_tArr[k] + step,
                _ruku4_x[k,:] + step * f3)

        xnew = step * (f1 + 2 * f2 + 2 * f3 + f4) / 6
        if type(xnew) is np.ndarray:
            xnew = xnew[-1]

        _ruku4_x[k + 1,:] = _ruku4_x[k,:] + xnew

    return _ruku4_tArr, _ruku4_x


def ruku4ErrorEstimation(f, x0, t0, tf, h, obtainedX):
    """Estimate the error for using ruku4 with a given step.

    Arguments:
        f:  Function originally used for ruku4
        x0: Initial value originally used for ruku4
        t0: Initial time originally used for ruku4
        tf: Final time originally used for ruku4
        h:  Step originally used for ruku4
        obtainedX: Position values obtained from ruku4

    Returns:
        Absolute value of the estimated error.

    Raises:
        Nothing
    """
        
    # Error estimation is made recalculating the same function but with
    # half the original step.
    halfStep = h / 2

    # There are cases where obtainedX's size has an odd value, whenever
    # this happens, we need to add two extra values to the error array, so the
    # last one can be compared without raising an IndexError exception.
    if len(obtainedX) & 1:
        tf += 2 * halfStep

    _, xHalfStep = ruku4(f, x0, t0, tf, halfStep)

    error = abs((obtainedX[-1][0] - xHalfStep[-1][0])) / 15

    return error


def testRuku4():
    """Compare ruku4 function using ODEs with and without a kwnown solution.

    In the case where the solution of the ODE is kwnown, ruku4 return values
    are directely compared against the solution equation, at the same points.
    On the other hand, those ODEs which have not a kwnown solution (or it has
    not been calculated), this are compared against RK45 implementation in
    scipy.
    """

    # Some example functions were taken from:
    # http://epsem.upc.edu/~fpq/minas/pr-sol/prob-edos-n-solu
    #
    # Or were given during the corresponding lecture.

    # Step is calculated as the second element of 'interval'/'step_divisor'
    
    # ODEs whose solution is trivial or it's well kwnown.
    odesWithSolution = (
        {
            'function': ['cos(t)', lambda t, x: np.cos(t)],
            'solution': lambda t: np.sin(t),
            'interval': [0, 100],
            'step_divisor': 100000,
            'x0': np.zeros(1)
        },
        {
            'function': ['sin(t)', lambda t, x: np.sin(t)],
            'solution': lambda t: (-1) * np.cos(t) + 1,
            'interval': [0, 80],
            'step_divisor': 10000,
            'x0': np.zeros(1)
        },
        {
            # This function is taken from an RC circuit.
            # dx = (A*cos(w*t)-x) / (R*C)
            # Where, A = 1.0; w = 2*pi*1000; R = 1e3 and C = 1e-6
            'function': ['(1.0*cos(2*pi*1000*t)-x)/(1e3*1e-6)',
                         lambda t, x: ((np.cos(np.pi * 2000 * t) - x) * 1000)],
            'solution': lambda t: (((-1)*np.exp(-t/(1e-3)) \
                                    + np.cos(2000*np.pi*t) \
                                    + 2.0*np.pi*np.sin(2000*np.pi*t)) \
                                   / (1+4.0*(np.pi)**2)),
            'interval': [0, 0.005],
            'step_divisor': 100000,
            'x0': np.zeros(1)
        },
    )
    
    # ODEs without an analytic solution to test against RK45.
    odesWithNoSolution = (
        {
            'function': ['5*e^(-t)+0.5*e^(-2t)-e^(-2t)*sin(e^t)',
                         lambda t, x:
                         5 * np.exp(-t) + np.exp(-2 * t) * (0.5 \
                             - np.sin(np.exp(t)))],
            'interval': [-5, 5],
            'step_divisor': 10000,
            'x0': np.zeros(1)
        },
        {
            'function': ['1.64*e^(-t)-5.45te^(-t)+e^(-t)*(t^2)*(ln(t)/2-3/4)',
                         lambda t, x: np.exp(-t) * (1.64 \
                                - 5.45 * t + t * t * (np.log(t) / 2 - 3 / 4))],
            'interval': [0.01, 10],
            'step_divisor': 10000,
            'x0': np.zeros(1)
        },
    )

    print('=' * 60, "\nTesting ruku4()...\n", '-' * 59)

    # ODEs whose solution is trivial or it's well kwnown.
    for fun in odesWithSolution:
        fname = fun['function'][0]
        fsol = fun['solution']
        x0 = fun['x0']
        t0 = fun['interval'][0]
        tf = fun['interval'][1]
        step = fun['interval'][1] / fun['step_divisor']


        print('Comparing:')
        print(f'\tf(t, x) = {fname}')
        print(f'\tIn [{t0} ; {tf}], using a step = {step} and x0 = {x0}')

        tRuku4, xRuku4 = ruku4(fun['function'][1], x0, t0, tf, step)
        solution = fsol(tRuku4)
        
        equalX = _compareOutputs(xRuku4, solution)

        if not equalX:
            print('X FAIL X')
            print('Errors were found while comparing both functions.')
        else:
            print('-> Pass <-')

    # ODEs without an analytic solution to test against RK45.
    for fun in odesWithNoSolution:
        fname = fun['function'][0]
        x0 = fun['x0']
        t0 = fun['interval'][0]
        tf = fun['interval'][1]
        step = fun['interval'][1] / fun['step_divisor']

        print('Comparing:')
        print(f'\tf(t, x) = {fname}')
        print(f'\tIn [{t0} ; {tf}], using a step = {step} and x0 = {x0}')

        tRuku4, xRuku4 = ruku4(fun['function'][1], x0, t0, tf, step)

        rk45 = solve_ivp(fun['function'][1], [t0, tf], x0, method='RK45',
                         t_eval=tRuku4, rtol=1e-13, atol=1e-14)

        equalX = _compareOutputs(xRuku4, rk45.y.T)

        if not equalX:
            print('X FAIL X')
            print('Errors were found while comparing both functions.')
        else:
            print('-> Pass <-')


def _compareOutputs(ruku4_x, rk45_x):
    """Compare the output between our ruku4 and an already implemented RK45."""

    allSimilar = True
    for i in range(ruku4_x.size):
        try:
            ans = math.isclose(ruku4_x[i], rk45_x[i],
                               rel_tol=1e-05, abs_tol=1e-07)
            if not ans:
                print(f'False: {ruku4_x[i]} || {rk45_x[i]}')
                allSimilar = False

        except IndexError:
            print('The computed value was too large.',
                  'Try reducing the interval for this function.')
            return False

    return allSimilar


if __name__ == "__main__":
    print("Running tests...")
    initialTime = time.time_ns()

    testRuku4()

    finalTime = time.time_ns()

    print("\nTests ran in: %f s (%d ns)\n" % ((finalTime - initialTime) \
                                            * 10 ** (-9),
                                            finalTime - initialTime))
