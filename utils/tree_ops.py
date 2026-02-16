import jax
import jax.numpy as jnp
from typing import Callable, Any, Sequence, Optional, Union


def binary_tree_op(op: Callable[[Any, Any], Any], operand1: Any, operand2: Any) -> Any:
    """
    Applies a binary operation between two operands, which can be either PyTrees or JAX Arrays.
    Automatically reshapes the lower-rank array to broadcast against the higher-rank array
    (assuming leading dimensions match, e.g., (N,) vs (N, D)).

    Args:
        op: Function taking (x, y) -> result (e.g., operator.mul, jnp.add).
        operand1: PyTree or Array.
        operand2: PyTree or Array.

    Returns:
        Resulting PyTree or Array.
    """
    def _apply_leaf_broadcast(l1, l2):
        ndim1 = getattr(l1, 'ndim', 0)
        ndim2 = getattr(l2, 'ndim', 0)
        
        diff = ndim1 - ndim2
        
        if diff > 0:
            # l1 has more dims (e.g., N, 2). Reshape l2 (e.g., N) to (N, 1)
            new_shape = l2.shape + (1,) * diff
            l2 = l2.reshape(new_shape)
        elif diff < 0:
            # l2 has more dims. Reshape l1 to match
            new_shape = l1.shape + (1,) * abs(diff)
            l1 = l1.reshape(new_shape)
            
        return op(l1, l2)

    # anything with '.ndim' (Array/Tracer) is a Leaf, else it's a tree
    is_array1 = hasattr(operand1, 'ndim')
    is_array2 = hasattr(operand2, 'ndim')


    # Case: Tree vs Tree (structure must match)
    if not is_array1 and not is_array2:
        return jax.tree.map(_apply_leaf_broadcast, operand1, operand2)
    # Case: Tree vs Array (Broadcast array to tree)
    elif not is_array1 and is_array2:
        return jax.tree.map(lambda x: _apply_leaf_broadcast(x, operand2), operand1)
    # Case: Array vs Tree (Broadcast array to tree)
    elif is_array1 and not is_array2:
        return jax.tree.map(lambda y: _apply_leaf_broadcast(operand1, y), operand2)
    # Case: Array vs Array
    else:
        return _apply_leaf_broadcast(operand1, operand2)
    

def accumulate_weighted_trees(trees: Sequence[Optional[Any]], 
                              weights: Sequence[Union[float, jnp.array]]) -> Optional[Any]:
    """
    Computes a weighted sum of PyTrees: sum(tree * weight for tree, weight in zip(trees, weights)).
    Args:
        trees: A sequence (list/tuple) of PyTrees (or None).
        weights: A sequence of scalars corresponding to each tree.
        
    Returns:
        A new PyTree with the accumulated values, or None if no valid trees were provided.
    """
    assert len(trees) == len(weights)
    accumulated = None
    for tree, weight in zip(trees, weights):
        if tree is None:
            continue
        weighted_tree = jax.tree.map(lambda x: weight * x, tree)
        if accumulated is None:
            accumulated = weighted_tree
        else:
            accumulated = jax.tree.map(lambda x, y: x + y, accumulated, weighted_tree)
    return accumulated