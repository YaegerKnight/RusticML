"""
description: This class represents the core tensor object that is the foundation
to represent and operate all data
"""


class Tensor(object):

    def __init__(self, value, children, gradient, name  ) -> None:
        self.value = value
        self.children = ()
        self.gradient = 0.0
        self.