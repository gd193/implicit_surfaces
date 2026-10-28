// Resident native-layout grids to upstream VTK-m CUDA contour worklets.
#include <vtkm/cont/ArrayHandleBasic.h>
#include <vtkm/cont/ArrayHandleUniformPointCoordinates.h>
#include <vtkm/cont/CellSetStructured.h>
#include <vtkm/cont/CoordinateSystem.h>
#include <vtkm/cont/RuntimeDeviceTracker.h>
#include <vtkm/cont/cuda/DeviceAdapterCuda.h>
#include <vtkm/filter/contour/worklet/contour/MarchingCells.h>
#include <vtkm/filter/contour/worklet/contour/FlyingEdges.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace {
thread_local std::string error;
struct Result {
  vtkm::cont::ArrayHandle<vtkm::Vec3f> vertices;
  vtkm::cont::CellSetSingleType<> cells;
  vtkm::cont::ArrayHandle<vtkm::Id2> edges;
};
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
}
extern "C" const char* vtkm_mc_error() {return error.c_str();}
extern "C" void* vtkm_mc_run(float* grid, int nx, int ny, int nz, float iso, int flyingEdges, int captureEdges,
  const float* lower, const float* spacing) {
  try {
    error.clear();
    if (!grid || nx < 2 || ny < 2 || nz < 2) throw std::runtime_error("invalid grid");
    check(cudaSetDevice(0));
    vtkm::cont::GetRuntimeDeviceTracker().ForceDevice(vtkm::cont::DeviceAdapterTagCuda{});
    auto result = std::make_unique<Result>();
    vtkm::cont::CellSetStructured<3> cells;
    const vtkm::Id3 dims{nx, ny, nz};
    cells.SetPointDimensions(dims);
    vtkm::cont::ArrayHandleBasic<float> rawValues(grid, vtkm::Id(nx)*ny*nz,
      vtkm::cont::DeviceAdapterTagCuda{}, [](void*){});
    vtkm::cont::ArrayHandleUniformPointCoordinates coords(dims,
      {lower[0],lower[1],lower[2]}, {spacing[0],spacing[1],spacing[2]});
    vtkm::cont::ArrayHandle<vtkm::Vec3f> normals;
    vtkm::worklet::contour::CommonState state(true);
    state.GenerateNormals = false;
    auto run = [&](const auto& field) {
      if (flyingEdges) {
        return vtkm::worklet::flying_edges::execute(cells,
          vtkm::cont::CoordinateSystem("coords", coords), std::vector<float>{iso},
          field, result->vertices, normals, state);
      }
      return vtkm::worklet::marching_cells::execute<3>(cells, coords,
        std::vector<float>{iso}, field, result->vertices, normals, state);
    };
    result->cells = run(rawValues);
    if (captureEdges) result->edges = state.InterpolationEdgeIds;
    check(cudaStreamSynchronize(cudaStreamPerThread));
    return result.release();
  } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
extern "C" int vtkm_mc_info(void* handle, void** vertices, void** faces, int* nv, int* nt) {
  try {
    auto& r = *static_cast<Result*>(handle);
    vtkm::cont::Token token;
    auto vp = r.vertices.PrepareForInput(vtkm::cont::DeviceAdapterTagCuda{}, token);
    auto connectivity = r.cells.GetConnectivityArray(vtkm::TopologyElementTagCell{}, vtkm::TopologyElementTagPoint{});
    auto fp = connectivity.PrepareForInput(vtkm::cont::DeviceAdapterTagCuda{}, token);
    *vertices = const_cast<vtkm::Vec3f*>(vp.GetIteratorBegin());
    *faces = const_cast<vtkm::Id*>(fp.GetIteratorBegin());
    *nv = int(r.vertices.GetNumberOfValues());
    *nt = int(r.cells.GetNumberOfCells());
    return 0;
  } catch (const std::exception& e) {error=e.what(); return -1;}
}
extern "C" void vtkm_mc_free(void* handle) {delete static_cast<Result*>(handle);}
extern "C" int vtkm_mc_copy(void* handle, float* vertices, int* faces) {
  void *vp, *fp; int nv, nt;
  if (vtkm_mc_info(handle,&vp,&fp,&nv,&nt)) return -1;
  try {
    check(cudaMemcpy(vertices,vp,size_t(nv)*3*sizeof(float),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(faces,fp,size_t(nt)*3*sizeof(int),cudaMemcpyDeviceToHost));
    return 0;
  } catch(const std::exception& e) {error=e.what(); return -1;}
}

extern "C" int vtkm_mc_copy_edges(void* handle, int* edges) {
  try {
    auto& r=*static_cast<Result*>(handle);
    if (r.edges.GetNumberOfValues()!=r.vertices.GetNumberOfValues())
      throw std::runtime_error("edge metadata was not retained; use capture_edges=True");
    vtkm::cont::Token token;
    auto portal=r.edges.PrepareForInput(vtkm::cont::DeviceAdapterTagCuda{},token);
    check(cudaMemcpy(edges,portal.GetIteratorBegin(),size_t(r.vertices.GetNumberOfValues())*2*sizeof(int),cudaMemcpyDeviceToHost));
    return 0;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
