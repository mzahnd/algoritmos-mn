#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def _plot(plotTitle='My Graph', xAxisLabel='X', yAxisLabel='Y', *argv):
    """Display multiple on a single plot window.

    Arguments:
        plotTitle: Self descriptive.
        xAxisLabel: Self descriptive.
        yAxisLabel: Self descriptive.
        argv: Recieves one or more tuples (one for each function) with the
            points to plot and, if desired, a legend.
            
            The tuples **MUST** have the following structure:
                (xAxisPoints, yAxisPoints, 'FunctionLegend')

                Where xAxisPoints and yAxisPoints are lists or tuples and
                'FunctionLegend' is a string.

                When no function legend is desired for one or more functions,
                the third argument must be left as an empty string: ''.
                Resulting in:
                    (xAxisPoints, yAxisPoints, '')

    Returns:
        Nothing.
        Displays the plot on the screen.

    Raises:
        RuntimeError.
    """
    noFunctionLegend = True
    
    print("Plotting...")
    _, axes = plt.subplots()

    for plotData in argv:
        if type(plotData) is not tuple:
            raise RuntimeError('Each argv must be a three element tuple.')
        elif len(plotData) != 3:
            raise RuntimeError('Each argv must be a three element tuple.')

        if type(plotData[0]) is (not list or not tuple):
            raise RuntimeError('Please provide a list or a tuple for the X axis data.')
        elif type(plotData[1]) is (not list or not tuple):
            raise RuntimeError('Please provide a list or a tuple for the Y axis data.')
        elif (len(plotData[0]) != len(plotData[1])):
            raise RuntimeError('X and Y axis must have the same dimension.')
        elif type(plotData[2]) is not str:
            raise RuntimeError('Please provide a string for the function legend.')

        xAxis = plotData[0]
        yAxis = plotData[1]
        functionLegend = plotData[2]

        axes.plot(xAxis, yAxis[:], label=functionLegend)

        if not functionLegend == '':
            noFunctionLegend = False
    
    plt.title(plotTitle)
    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel)
    
    if not noFunctionLegend:
        axes.legend()
    

    print("Plots ready.")
    print('Press enter to show both plots.')
    input()

    plt.show()

if __name__ == "__main__":
    # Example plot
    points1 = [1, 2, 3, 4]
    points2 = (1, 4, 9, 16)

    points3 = points2
    points4 = points1

    _plot('My Graph', 'X', 'Y', 
        (points1, points2, 'Graph'), (points3, points4, 'Graph 2') )