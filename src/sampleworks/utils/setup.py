"""
Various utility functions useful when initializing classes and modules from Sampleworks.
"""


class DotDict(dict):
    """
    A dictionary subclass that supports attribute-style access.

    This class allows you to access dictionary keys as if they were attributes.
    For example, instead of writing `d['key']`, you can write `d.key`.

    Example usage:
        d = DotDict()
        d.key = 'value'
        print(d.key)  # Output: value
        print(d['key'])  # Output: value

    From: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
