#include <fcpw/fcpw_gpu.h>
#include "fcpw_timing.h"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace fcpw;

template<class T> std::vector<T> read(const std::string& path) {
    const auto size = std::filesystem::file_size(path);
    if (size % sizeof(T)) throw std::runtime_error("invalid array: " + path);
    std::vector<T> values(size/sizeof(T));
    if (!std::ifstream(path, std::ios::binary).read((char*)values.data(), size))
        throw std::runtime_error("cannot read " + path);
    return values;
}

int main(int argc, char** argv) try {
    if (argc != 8) throw std::runtime_error("usage: fcpw_queries SOURCE FIXTURE MODE BLOCK LAUNCHES REPEATS PREFIX");
    const std::string source = argv[1], fixture = argv[2], mode = argv[3], prefix = argv[7];
    const int block = std::stoi(argv[4]), launches = std::stoi(argv[5]), repeats = std::stoi(argv[6]);
    if ((block != 64 && block != 128 && block != 256) || launches < 1 || repeats < 1)
        throw std::runtime_error("invalid timing configuration");
    if (mode != "nearest" && mode != "silhouette" && mode != "first" && mode != "any")
        throw std::runtime_error("invalid query mode");
    const bool silhouette = mode == "silhouette";
    auto xy = read<float>(fixture + "/vertices.f32");
    auto indices = read<int>(fixture + "/indices.i32");
    auto points = read<float>(fixture + "/points.f32");
    auto directions = read<float>(fixture + "/directions.f32");
    if (xy.size()%2 || indices.size()%2 || points.size()%2 || points.size()!=directions.size())
        throw std::runtime_error("expected packed 2D arrays");
    Scene<2> scene;
    scene.setObjectCount(1);
    std::vector<Vector<2>> vertices(xy.size()/2);
    std::vector<Vector2i> segments(indices.size()/2);
    for (size_t i=0; i<vertices.size(); ++i) vertices[i] = Vector<2>(xy[2*i], xy[2*i+1]);
    for (size_t i=0; i<segments.size(); ++i) segments[i] = Vector2i(indices[2*i], indices[2*i+1]);
    scene.setObjectVertices(vertices, 0);
    scene.setObjectLineSegments(segments, 0);
    if (silhouette) scene.computeSilhouettes();
    scene.build(AggregateType::Bvh_SurfaceArea, false);

    const std::string shaders = source + "/include/fcpw/gpu";
    auto bytes = read<char>(shaders + "/bvh.cs.slang");
    std::string entry(bytes.begin(), bytes.end());
    for (size_t at=0; (at=entry.find("numthreads(256,", at))!=std::string::npos; at+=15)
        entry.replace(at, 15, "numthreads(" + std::to_string(block) + ",");
    std::filesystem::create_directories(prefix + "_shader");
    const std::string shaderPath = prefix + "_shader/bvh.cs.slang";
    std::ofstream(shaderPath) << entry;
    GPUContext context;
    context.searchPaths = {shaders};
    context.macros = {{"FCPW_BVH_TYPE", silhouette ? "3" : "1"}};
    context.initDevice(DeviceType::CUDA);
    std::vector<GPUModule> libraries(1);
    libraries[0].load(context, shaders + "/fcpw.slang");
    GPUModule module;
    module.load(context, shaderPath);
    ComputeShader shader;
    shader.loadProgram(context, module, libraries, silhouette ? "closestSilhouettePoint" :
                       mode == "nearest" ? "closestPoint" : "rayIntersection");
    GPUBvhBuffers bvh;
    bvh.allocate<2>(context, scene.getSceneData(), true, silhouette, true, false);
    if (silhouette) {
        std::vector<GPUVertex> features;
        bvh.silhouettes.read<GPUVertex>(context, features);
        uint32_t size = 0;
        for (const auto& feature : features) size = std::max(size, feature.index+1);
        std::vector<float> positions(2*size);
        for (const auto& feature : features) { positions[2*feature.index]=feature.p.x; positions[2*feature.index+1]=feature.p.y; }
        std::ofstream(prefix+".features", std::ios::binary).write((char*)positions.data(), positions.size()*sizeof(float));
    }
    auto bind = [&](const ComputeShader& target, const ShaderCursor& cursor) {
        ComPtr<IShaderObject> object = target.createShaderObject(context, bvh.getReflectionType());
        ShaderCursor bindings(object);
        bvh.setResources(bindings, false);
        cursor.getPath("gBvh").setObject(object);
    };
    const size_t count = points.size()/2;
    auto run = [&](auto& query) {
        auto timing = timeKernelOnly(context, shader, query, bind, countThreadGroups(count, block, false), launches, repeats);
        std::ofstream samples(prefix + ".ms");
        samples.precision(12);
        for (double ms : timing.samples_ms) samples << ms << '\n';
        std::vector<GPUInteraction> results;
        query.read(context, results);
        std::ofstream output(prefix + ".bin", std::ios::binary);
        for (auto& hit : results) {
            float record[] = {hit.p.x, hit.p.y, hit.d};
            int id = hit.index == FCPW_GPU_UINT_MAX ? -1 : int(hit.index);
            output.write((char*)record, sizeof(record));
            output.write((char*)&id, sizeof(id));
        }
        std::ofstream(prefix + ".json") << "{\"registers\":" << timing.regs
            << ",\"local_bytes\":" << timing.local_bytes << ",\"captured_kernels\":" << timing.captured_kernels << "}\n";
    };
    if (mode == "first" || mode == "any") {
        std::vector<GPURay> rays(count);
        for (size_t i=0; i<count; ++i) rays[i] = GPURay({points[2*i],points[2*i+1],0}, {directions[2*i],directions[2*i+1],0}, 10.f);
        GPURunRayIntersectionQuery query;
        query.allocate(context, rays);
        query.checkForOcclusion = mode == "any";
        run(query);
    } else {
        std::vector<GPUBoundingSphere> spheres(count);
        for (size_t i=0; i<count; ++i) spheres[i] = GPUBoundingSphere({points[2*i],points[2*i+1],0}, .75f*.75f);
        if (silhouette) {
            GPURunClosestSilhouettePointQuery query;
            query.allocate(context, spheres, std::vector<uint32_t>(count, 0));
            query.precision = query.squaredMinRadius = 0;
            run(query);
        } else {
            GPURunClosestPointQuery query;
            query.allocate(context, spheres);
            run(query);
        }
    }
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
