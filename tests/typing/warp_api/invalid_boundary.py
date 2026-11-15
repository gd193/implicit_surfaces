"""Core packing must preserve the adapter's declared host schema."""
import warp as wp

from surfaces.explicit.accelerators.bvh.warp import BoundaryAdapter, BoundaryCore, BoundaryCoreData, TreeData
from tests.typing.warp_api.query_extension import FeatureArrays, FeatureData, MetadataArrays, prepare_core
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalData, ExternalItem


def wrong_pack(core: BoundaryCoreData, data: ExternalArrays, tree: TreeData,
               geometry: ExternalArrays) -> MetadataArrays:
    return MetadataArrays(geometry.positions)


BoundaryAdapter[ExternalArrays, ExternalData, ExternalItem, wp.vec3,
                BoundaryCoreData, BoundaryCore, FeatureArrays, FeatureData, ExternalItem](
    prepare=prepare_core, pack_core=wrong_pack)  # expect: reportArgumentType
