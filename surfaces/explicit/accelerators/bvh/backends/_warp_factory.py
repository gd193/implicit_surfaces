"""Typed array entry points for factory-built Warp queries."""
from __future__ import annotations

from dataclasses import dataclass
import linecache
from itertools import count

import warp as wp

from ._warp_jax import batched_kernel
from ._warp_specs import STACK_CAPACITY
from ._warp_tree import DeviceTree


_names = count()


@dataclass(frozen=True)
class BoundQuery:
    kernel: object
    primitive: object
    num_query_args: int
    num_outputs: int
    block_dim: int

    def launch(self, tree, geometry, shape, *args, output_shapes=None):
        call = batched_kernel(self.kernel, self.num_query_args, self.num_outputs, self.block_dim, output_shapes)
        values = call(*args, *struct_arrays(tree, DeviceTree), *struct_arrays(geometry, self.primitive.data_type))
        return tuple(v.reshape(shape+v.shape[1:]) for v in values)


def array_kernel(name, arguments, body, bindings):
    """Materialize only the flat FFI signature; algorithms stay Warp functions."""
    declarations = ", ".join(f"{key}: T{i}" for i, (key, _) in enumerate(arguments))
    source = f"def {name}({declarations}):\n" + "\n".join("    " + line for line in body) + "\n"
    filename = f"<warp_bvh_{name}_{next(_names)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": __name__, "wp": wp, **bindings}
    namespace.update({f"T{i}": value for i, (_, value) in enumerate(arguments)})
    exec(compile(source, filename, "exec"), namespace)
    return wp.kernel(module="unique", enable_backward=False)(namespace[name])


def struct_arguments(prefix, struct_type):
    if any(not isinstance(var.type, wp.array) for var in struct_type.vars.values()):
        raise TypeError("payload struct fields must be Warp arrays")
    return [(f"{prefix}_{name}", var.type) for name, var in struct_type.vars.items()]


def struct_body(prefix, struct_type, symbol):
    return [f"{prefix} = {symbol}()", *(
        f"{prefix}.{name} = {prefix}_{name}" for name in struct_type.vars)]


def struct_arrays(value, struct_type):
    return tuple(getattr(value, name) for name in struct_type.vars)


def native_call(native, aliases):
    return "wp.static(native)(" + ", ".join(aliases.get(a.label, a.label) for a in native.adj.args) + ")"


def native_entrypoint(name, native, spec):
    aliases = {"qi": "i", "cap": "wp.inf", "node_count": f"{spec.payload_fields[0]}.shape[0]"}
    arguments = []
    for arg in native.adj.args:
        if arg.label not in aliases:
            label = {"origin": "origins", "direction": "directions"}.get(arg.label, arg.label)
            is_array = isinstance(arg.type, wp.array)
            arguments.append((label, arg.type if is_array else wp.array(dtype=arg.type)))
            if not is_array:
                aliases[arg.label] = f"{label}[i]"
    return array_kernel(name, arguments, ["i = wp.tid()", native_call(native, aliases)], {"native": native})


def bind_query(query_type, name, primitive, inputs, outputs, body, bindings, block_dim,
               *, native=None, id_field="out_id", excluded="-1", **options):
    Geometry = primitive.data_type
    arguments = [*inputs, *struct_arguments("tree", DeviceTree), *struct_arguments("geometry", Geometry), *outputs]
    if native is None:
        body = [*struct_body("tree", DeviceTree, "Tree"), *struct_body("geometry", Geometry, "Geometry"), *body]
        bindings = {"Tree": DeviceTree, "Geometry": Geometry, **bindings}
    else:
        aliases = {name: f"tree_{name}" for name in DeviceTree.vars}
        aliases.update(qi="i", query="queries[i]", cap="caps[i]", origin="origins[i]",
            direction="directions[i]", min_t="minimum[i]", max_t="maximum[i]", exclude_face=excluded,
            node_count="tree_node_left.shape[0]", tri_v0="geometry_a", tri_e1="geometry_e1",
            tri_e2="geometry_e2", mesh_vertices="geometry_vertices", tri_indices="geometry_indices",
            out_primitive_id=id_field, out_hit="out_count")
        body, bindings = [native_call(native, aliases)], {"native": native}
    kernel = array_kernel(name, arguments, ["i = wp.tid()", *body], bindings)
    return query_type(kernel, primitive, len(inputs), len(outputs), block_dim, **options)


def is_native_triangle(primitive, stack_capacity):
    from ._warp_primitives import Primitive, Triangles

    return (stack_capacity == STACK_CAPACITY and type(primitive) is Primitive
            and primitive.name == "triangles" and primitive == Triangles())
