"""Typed array entry points for factory-built Warp queries."""
from __future__ import annotations

from dataclasses import dataclass, field
import linecache
from itertools import count

import warp as wp

from ._warp_jax import batched_kernel
from ._warp_traversal import QUERY_BUILD_OPTIONS, load_array, store_array
from ._warp_tree import DeviceTree


_names = count()


@dataclass(frozen=True)
class BoundQuery:
    kernel: object
    primitive: object
    num_query_args: int
    num_outputs: int
    block_dim: int
    group_types: tuple = field(default=(), kw_only=True)

    def launch(self, tree, geometry, shape, *args, output_shapes=None, metadata=()):
        if len(metadata) != len(self.group_types):
            raise ValueError("query metadata does not match its static schema")
        call = batched_kernel(self.kernel, self.num_query_args, self.num_outputs, self.block_dim, output_shapes)
        extra = tuple(a for value, schema in zip(metadata, self.group_types, strict=True)
                      for a in struct_arrays(value, schema))
        values = call(*args, *struct_arrays(tree, DeviceTree), *struct_arrays(geometry, self.primitive.data_type), *extra)
        return tuple(v.reshape(shape+v.shape[1:]) for v in values)


def array_kernel(name, arguments, body, bindings):
    """Materialize only the flat FFI signature; algorithms stay Warp functions."""
    declarations = ", ".join(f"{key}: T{i}" for i, (key, _) in enumerate(arguments))
    source = f"def {name}({declarations}):\n" + "\n".join("    " + line for line in body) + "\n"
    filename = f"<warp_bvh_{name}_{next(_names)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": __name__, "wp": wp, "load_array": load_array, "store_array": store_array, **bindings}
    namespace.update({f"T{i}": value for i, (_, value) in enumerate(arguments)})
    exec(compile(source, filename, "exec"), namespace)
    kernel = wp.kernel(module="unique", enable_backward=False)(namespace[name])
    wp.set_module_options({"extra_build_options": QUERY_BUILD_OPTIONS}, module=kernel.module)
    return kernel


def struct_arguments(prefix, struct_type):
    if any(not isinstance(var.type, wp.array) for var in struct_type.vars.values()):
        raise TypeError("payload struct fields must be Warp arrays")
    return [(f"{prefix}_{name}", var.type) for name, var in struct_type.vars.items()]


def struct_body(prefix, struct_type, symbol):
    return [f"{prefix} = {symbol}()", *(
        f"{prefix}.{name} = {prefix}_{name}" for name in struct_type.vars)]


def struct_arrays(value, struct_type):
    return tuple(getattr(value, name) for name in struct_type.vars)


def bind_query(query_type, name, primitive, inputs, outputs, body, bindings, block_dim, *, groups=(), **options):
    schemas = (("tree", DeviceTree), ("geometry", primitive.data_type), *groups)
    arguments = [*inputs, *(arg for prefix, schema in schemas for arg in struct_arguments(prefix, schema)), *outputs]
    body = [*(line for prefix, schema in schemas for line in struct_body(prefix, schema, prefix+"_type")), *body]
    bindings = {**{prefix+"_type": schema for prefix, schema in schemas}, **bindings}
    kernel = array_kernel(name, arguments, ["i = wp.tid()", *body], bindings)
    return query_type(kernel, primitive, len(inputs), len(outputs), block_dim,
                      group_types=tuple(schema for _, schema in groups), **options)
