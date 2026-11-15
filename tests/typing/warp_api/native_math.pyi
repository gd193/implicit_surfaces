from surfaces.explicit.accelerators.bvh.warp import WarpFunction
import warp as wp

squared_distance: WarpFunction[[wp.vec3, wp.vec3], float]
vector_x: WarpFunction[[wp.vec3], float]
