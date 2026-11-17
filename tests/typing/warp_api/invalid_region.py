"""Reject region callbacks for other primitives and swapped metadata."""
import warp as wp
from surfaces.explicit.accelerators.bvh.warp import (
    DeviceTree, Points, PointsData3, make_region, triangle_region_policy,
)
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalData
from tests.typing.warp_api.query_extension import ExternalMetadata
from tests.typing.warp_api.region_valid import external, classified


def invalid(tree: DeviceTree, points: PointsData3, data: ExternalData,
            host: ExternalArrays, metadata: ExternalMetadata, p: wp.vec3) -> None:
    make_region(Points(), triangle_region_policy())  # expect: reportArgumentType
    external.device(p, p, tree, points)  # expect: reportArgumentType
    external.device(wp.vec2(), p, tree, data)  # expect: reportArgumentType
    classified.device(p, p, tree, data, points)  # expect: reportArgumentType
    classified(tree, host, metadata, 0.0, 0.0)  # expect: reportArgumentType
