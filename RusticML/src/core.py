"""
description: This class represents the core tensor object that is the foundation
to represent and operate all data
"""
class Tensor(object):

    def __init__(self, value, parents=(), gradient=0.0, operation='', label=''  ) -> None:
        self.value = value
        self._backward = lambda: None
        self.parents = set(parents)
        self.gradient = gradient
        self.operation = operation
        self.label = label
    
    def __repr__(self) -> str:
        return f"Tensor(value={self.value})"

    def __add__(self, operand):
        
        if not isinstance(operand, Tensor):
            operand = Tensor(value=operand)

        result = Tensor(self.value + operand.value,parents=(self, operand), operation='+')

        def _backward():
            self.gradient += 1.0* result.gradient
            operand.gradient += 1.0 * result.gradient
        result._backward = _backward

        return result
    
    def __mul__(self, operand):

        if not isinstance(operand, Tensor):
            operand = Tensor(value=operand)
        
        result = Tensor(value = (self.value * operand.value), parents=(self, operand), operation='*')

        def _backward():
            self.gradient += operand.value * result.gradient
            operand.gradient += self.value * result.gradient
        result._backward = _backward

        return result

    def __rmul__(self, operand): 
        return self * operand

    def __sub__(self, operand):
        return self + (-operand) 

    def __radd__(self, operand):
        return self + operand

    def __pow__(self, operand):

        result = Tensor(value=(self.value ** operand), parents=(self,), operation='^')

        def _backward():
            self.gradient += operand * (self.value ** (operand - 1)) * result.gradient
        result._backward = _backward

        return result
    
    def backward(self):

        node_list = []
        visited_nodes = set()
        
        def topological_sort(root):
            if root not in visited_nodes:
                visited_nodes.add(root)

                for node in root.parents:
                    topological_sort(node)
            node_list.append(root)
        
        # This builds the list of nodes from initial layers to the final
        topological_sort(self)

        self.gradient = 1.0
        
        # The list is reversed because we want the gradient to flow from the last layer to the first layer
        for node in reversed(node_list):
            node._backward()