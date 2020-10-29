#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def _closest(lst, number):
    """Find the closest number in a list.
    
    Arguments:
        lst: List to look into.
        number: Desired number.
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - number))]
