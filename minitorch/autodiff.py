from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, Dict


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals1 = list(vals)
    vals2 = list(vals)

    vals1[arg] += epsilon
    vals2[arg] -= epsilon

    return (f(*vals1) - f(*vals2)) / (2 * epsilon)
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate derivative"""
        ...

    @property
    def unique_id(self) -> int:
        """Unique Id"""
        ...

    def is_leaf(self) -> bool:
        """Is_leaf"""
        ...

    def is_constant(self) -> bool:
        """Is constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Parents"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Chain rule"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    # visited = set()
    # partial_order = []

    # def dfs(v: Variable) -> None:
    #     if v.unique_id in visited:
    #         return
    #     visited.add(v.unique_id)
    #     print(v.parents)
    #     for parent in v.parents:
    #         dfs(parent)
    #     partial_order.append(v)

    # dfs(variable)
    # return reversed(partial_order)
    traversed: Dict[int, Variable] = dict()

    def traverse(variable: Variable) -> None:
        if variable.unique_id in traversed.keys() or variable.is_constant():
            return
        if not variable.is_leaf():
            for parent in variable.parents:
                traverse(parent)

        traversed[variable.unique_id] = variable

    traverse(variable)
    return list(traversed.values())[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
    None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    ordered_vars = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in ordered_vars:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            for parent, local_deriv in var.chain_rule(derivatives[var.unique_id]):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = local_deriv
                else:
                    derivatives[parent.unique_id] += local_deriv

    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Saved tensors"""
        return self.saved_values
