"""
Class representing nodes in the linear bound propagation computational graph.

The following example will explain the bound propagation logic used in the subclasses of the Node class. Say we have
three nodes in the computational graph, Node0, Node1, and Node2, where Node1 and Node2 are children of Node0.
Mathematically, the nodes represent the following relationships between the input x0 and output x2:

    (Node0)    l0 <= x0 <= u0
    (Node1)    l1 <= Omega1 @ x0 + theta1 <= x1 <= Lambda1 @ x0 + delta1 <= u1
    (Node2)    l2 <= Omega2 @ x1 + theta2 <= x2 <= Lambda2 @ x1 + delta2 <= u2

The aim of the bound propagation procedure is to compute the bounds on the output x2 in terms of the input x0. We do
this by starting with the bounds on the output of the last node (Node2) and substituting the bounds from the previous
node (Node1). For example, let's look at the j-th output of Node2:

    x2[j] <= Lambda2[j, :] @ x1 + delta2[j]
          = sum_i Lambda2[j, i] * x1[i] + delta2[j]
          <= sum_i (Lambda2[j, i] > 0) * Lambda2[j, i] * (Lambda1[i, :] @ x0 + theta1[i])       [Positive part]
             + sum_i (Lambda2[j, i] < 0) * Lambda2[j, i] * (Omega1[i, :] @ x0 + theta1[i])      [Negative part]
             + sum_i (0 in Lambda2[j, i]) * max(Lambda2[j, i] * [l2[i], u2[i]])                 [Undefined part]

Remember that the coefficients Lambda and Omega are intervals, so we need to consider the three cases of the sign of the
coefficient and all operations are done using interval arithmetic where appropriate. The same logic is applied to the
lower bound of x2, and the bounds on x0 are then concretized using interval arithmetic to get the final bounds on x2.

Each node class is focused on implementing this propagation logic for a specific type of operation. 
These implementations aim to exploit structure in the coefficients (such as diagonal, non-interval, batched) to make the
bound propagation more computationally efficient.
"""

from __future__ import annotations

import abc

import torch

from abstract_gradient_training.bounded_models._crown_bounds import IntervalBounds, LinearBounds


class Node(abc.ABC):
    """
    A node consisting of linear bounds on a function out_var = f(in_var).
    """

    def __init__(self):
        self.in_var: Node | None = None
        self.conc: IntervalBounds | None = None
        self._optimizable_params: list[torch.Tensor] = []

    @abc.abstractmethod
    def _backpropagate(self, backward_bounds: LinearBounds) -> None:
        """
        Extend the linear bounds passed in as an argument to include this node. Say the computational graph represented
        has the following structure:
            x0 -> x1 -> ... -> xn-1 -> xn -> xn+1 -> ... -> xN
        where
            - the current node (self) represents bounds L @ xn + d <= xn+1 <= O @ xn + t
            - the argument bounds (backward bounds) represents bounds L @ xn-1 + d <= xN <= O @ xn-1 + t
        then return linear bounds from xn-1 to xN of the form
            L @ xn-1 + d <= xN <= O @ xn-1 + t
        """

    @abc.abstractmethod
    def _init_backpropagation(self) -> LinearBounds:
        """
        Initialise the coefficients for performing backpropagation from this node.
        """

    def concretize(self) -> IntervalBounds:
        """
        Compute a tuple of concrete lower and upper bounds on the value of this node.
        """

        if self.conc is not None:
            return self.conc

        # get the linear relaxation of this node
        linear_bounds, input_conc = self.get_relaxation()
        self.conc = linear_bounds.concretize(input_conc)
        return self.conc

    def get_relaxation(self) -> tuple[LinearBounds, IntervalBounds]:
        """
        Return the linear relaxation of the node.

        Returns:
            LinearBounds: Linear relaxation of this node wrt the input node.
            IntervalBounds: Interval bounds on the input node.
        """
        assert not isinstance(self, InputNode), "Cannot get relaxation for input node"
        linear_bounds = self._init_backpropagation()
        cur = self.in_var
        assert cur is not None

        # backpropagate the bounds from the current node to its input node, until we reach a node without an input
        while cur.in_var is not None:
            cur._backpropagate(linear_bounds)
            cur = cur.in_var
        assert isinstance(cur, InputNode)
        return linear_bounds, cur.concretize()

    def clear_cached(self) -> None:
        """
        Clear the cached concretizations of this node and all its parents.
        """
        cur = self
        while cur is not None and not isinstance(cur, InputNode):
            cur.conc = None
            cur = cur.in_var

    def optimizable_parameters(self) -> list[torch.Tensor]:
        """
        Return a list of all optimizable parameters in the node and its children.
        """
        opt_params = []
        cur = self
        while cur is not None:
            opt_params.extend(cur._optimizable_params)
            cur = cur.in_var
        return opt_params


class InputNode(Node):
    """
    Sentinal node for the linear bound propagation computational graph representing the input to the network.
    """

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialise the Node with the given linear bounds relating the output to input of the node.
        Args:
            conc (IntervalBounds): Bounds on the concretization of this node
        """
        super().__init__()
        assert lb.shape == ub.shape
        if lb.dim() == 1:
            lb = lb.unsqueeze(0)
            ub = ub.unsqueeze(0)
        assert lb.dim() == 2
        self.lb = lb
        self.ub = ub

    def concretize(self) -> IntervalBounds:
        return IntervalBounds(self.lb, self.ub)

    def _backpropagate(self, backward_bounds: LinearBounds) -> None:
        raise ValueError("Cannot backpropagate from the input node")

    def _init_backpropagation(self) -> LinearBounds:
        raise ValueError("Cannot backpropagate from the input node")
