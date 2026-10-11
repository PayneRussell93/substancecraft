// Narrow C ABI over the Filament C++ API for the SubstanceCraft viewport.
//
// Every entry point catches C++ exceptions so they never unwind into Rust. Buffers handed to
// Filament are copied here and released by Filament's completion callbacks, so callers only
// need to keep their slices alive for the duration of the call.

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/SwapChain.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>

#include <utils/Entity.h>
#include <utils/EntityManager.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace filament;
using utils::Entity;
using utils::EntityManager;

namespace {

struct ScRenderer {
    Engine* engine = nullptr;
    SwapChain* swapChain = nullptr;
    Renderer* renderer = nullptr;
    Scene* scene = nullptr;
    View* view = nullptr;
    Camera* camera = nullptr;
    Entity cameraEntity;
    Entity sunEntity;
    IndirectLight* indirectLight = nullptr;
    Material* material = nullptr;
    MaterialInstance* materialInstance = nullptr;
    Texture* baseColorMap = nullptr;
    Entity meshEntity;
    VertexBuffer* vertexBuffer = nullptr;
    IndexBuffer* indexBuffer = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pendingReadbacks = 0;
};

void freeCallback(void* buffer, size_t, void*) {
    std::free(buffer);
}

void* copyBytes(void const* source, size_t size) {
    void* copy = std::malloc(size);
    if (copy) {
        std::memcpy(copy, source, size);
    }
    return copy;
}

float srgbToLinear(uint8_t value) {
    float const v = float(value) / 255.0f;
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

uint8_t linearToSrgb(float value) {
    float const v = value <= 0.0031308f ? value * 12.92f
                                        : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
    return uint8_t(std::lround(std::fmin(std::fmax(v, 0.0f), 1.0f) * 255.0f));
}

// 2x2 box filter in linear space (alpha averaged directly). Odd edges clamp to the last texel.
uint8_t* downsample(uint8_t const* source, uint32_t width, uint32_t height, uint32_t outWidth,
        uint32_t outHeight) {
    auto* out = static_cast<uint8_t*>(std::malloc(size_t(outWidth) * outHeight * 4));
    if (!out) {
        return nullptr;
    }
    for (uint32_t y = 0; y < outHeight; y++) {
        uint32_t const y0 = y * 2 < height ? y * 2 : height - 1;
        uint32_t const y1 = y * 2 + 1 < height ? y * 2 + 1 : height - 1;
        for (uint32_t x = 0; x < outWidth; x++) {
            uint32_t const x0 = x * 2 < width ? x * 2 : width - 1;
            uint32_t const x1 = x * 2 + 1 < width ? x * 2 + 1 : width - 1;
            uint8_t const* texels[4] = {
                    source + (size_t(y0) * width + x0) * 4, source + (size_t(y0) * width + x1) * 4,
                    source + (size_t(y1) * width + x0) * 4, source + (size_t(y1) * width + x1) * 4};
            uint8_t* target = out + (size_t(y) * outWidth + x) * 4;
            for (int channel = 0; channel < 3; channel++) {
                float sum = 0.0f;
                for (auto const* texel : texels) {
                    sum += srgbToLinear(texel[channel]);
                }
                target[channel] = linearToSrgb(sum * 0.25f);
            }
            target[3] = uint8_t(
                    (unsigned(texels[0][3]) + texels[1][3] + texels[2][3] + texels[3][3] + 2) / 4);
        }
    }
    return out;
}

void destroyMesh(ScRenderer* r) {
    if (!r->meshEntity.isNull()) {
        r->scene->remove(r->meshEntity);
        r->engine->destroy(r->meshEntity);
        EntityManager::get().destroy(r->meshEntity);
        r->meshEntity = {};
    }
    if (r->vertexBuffer) {
        r->engine->destroy(r->vertexBuffer);
        r->vertexBuffer = nullptr;
    }
    if (r->indexBuffer) {
        r->engine->destroy(r->indexBuffer);
        r->indexBuffer = nullptr;
    }
}

void destroyAll(ScRenderer* r) {
    if (!r->engine) {
        return;
    }
    Engine* engine = r->engine;
    destroyMesh(r);
    if (r->materialInstance) engine->destroy(r->materialInstance);
    if (r->baseColorMap) engine->destroy(r->baseColorMap);
    if (r->material) engine->destroy(r->material);
    if (r->indirectLight) engine->destroy(r->indirectLight);
    if (!r->sunEntity.isNull()) {
        engine->destroy(r->sunEntity);
        EntityManager::get().destroy(r->sunEntity);
    }
    if (!r->cameraEntity.isNull()) {
        engine->destroyCameraComponent(r->cameraEntity);
        EntityManager::get().destroy(r->cameraEntity);
    }
    if (r->view) engine->destroy(r->view);
    if (r->scene) engine->destroy(r->scene);
    if (r->renderer) engine->destroy(r->renderer);
    if (r->swapChain) engine->destroy(r->swapChain);
    Engine::destroy(&engine);
    r->engine = nullptr;
}

bool createSwapChain(ScRenderer* r, uint32_t width, uint32_t height) {
    if (r->swapChain) {
        r->engine->destroy(r->swapChain);
        r->swapChain = nullptr;
    }
    r->swapChain = r->engine->createSwapChain(width, height, SwapChain::CONFIG_READABLE);
    if (!r->swapChain) {
        return false;
    }
    r->width = width;
    r->height = height;
    r->view->setViewport({0, 0, width, height});
    return true;
}

typedef void (*sc_readback_fn)(void* user, uint8_t const* pixels, size_t len);

struct Readback {
    ScRenderer* renderer;
    sc_readback_fn callback;
    void* user;
};

void readbackCallback(void* buffer, size_t size, void* user) {
    auto* readback = static_cast<Readback*>(user);
    readback->renderer->pendingReadbacks--;
    readback->callback(readback->user, static_cast<uint8_t const*>(buffer), size);
    delete readback;
    std::free(buffer);
}

} // namespace

extern "C" {

ScRenderer* sc_renderer_create(uint8_t const* material, size_t material_len, uint32_t width,
        uint32_t height) {
    auto* r = new (std::nothrow) ScRenderer();
    if (!r) {
        return nullptr;
    }
    try {
        r->engine = Engine::create(Engine::Backend::OPENGL);
        if (!r->engine) {
            delete r;
            return nullptr;
        }
        r->renderer = r->engine->createRenderer();
        r->scene = r->engine->createScene();
        r->view = r->engine->createView();

        EntityManager& entities = EntityManager::get();
        r->cameraEntity = entities.create();
        r->camera = r->engine->createCamera(r->cameraEntity);
        r->view->setScene(r->scene);
        r->view->setCamera(r->camera);
        r->view->setPostProcessingEnabled(true);
        r->view->setShadowingEnabled(false);

        r->sunEntity = entities.create();
        LightManager::Builder(LightManager::Type::DIRECTIONAL)
                .color({1.0f, 0.96f, 0.9f})
                .intensity(80000.0f)
                .direction({-0.45f, -0.7f, -1.0f})
                .castShadows(false)
                .build(*r->engine, r->sunEntity);
        r->scene->addEntity(r->sunEntity);

        // Constant ambient term (SH band 0) so faces turned away from the sun stay readable.
        math::float3 const ambient[1] = {{0.9f, 0.92f, 1.0f}};
        r->indirectLight = IndirectLight::Builder()
                                   .irradiance(1, ambient)
                                   .intensity(20000.0f)
                                   .build(*r->engine);
        r->scene->setIndirectLight(r->indirectLight);

        r->material = Material::Builder().package(material, material_len).build(*r->engine);
        if (!r->material) {
            destroyAll(r);
            delete r;
            return nullptr;
        }
        r->materialInstance = r->material->createInstance();
        r->materialInstance->setParameter("roughness", 0.58f);
        r->materialInstance->setParameter("metallic", 0.0f);

        Renderer::ClearOptions clear;
        clear.clearColor = {0.025, 0.03, 0.035, 1.0};
        clear.clear = true;
        clear.discard = true;
        r->renderer->setClearOptions(clear);

        if (!createSwapChain(r, width, height)) {
            destroyAll(r);
            delete r;
            return nullptr;
        }
        return r;
    } catch (...) {
        destroyAll(r);
        delete r;
        return nullptr;
    }
}

void sc_renderer_destroy(ScRenderer* r) {
    if (!r) {
        return;
    }
    try {
        if (r->engine) {
            r->engine->flushAndWait();
            // Deliver outstanding readback callbacks so their Rust closures are released.
            r->engine->pumpMessageQueues();
        }
        destroyAll(r);
    } catch (...) {
    }
    delete r;
}

bool sc_renderer_resize(ScRenderer* r, uint32_t width, uint32_t height) {
    try {
        return createSwapChain(r, width, height);
    } catch (...) {
        return false;
    }
}

// Upload the Base Color map (sRGB RGBA8, rows top to bottom) with a full mip chain.
bool sc_renderer_set_base_color_map(ScRenderer* r, uint32_t width, uint32_t height,
        uint8_t const* rgba8) {
    try {
        uint8_t levels = 1;
        for (uint32_t extent = width > height ? width : height; extent > 1; extent >>= 1) {
            levels++;
        }
        Texture* texture = Texture::Builder()
                                   .width(width)
                                   .height(height)
                                   .levels(levels)
                                   .sampler(Texture::Sampler::SAMPLER_2D)
                                   .format(Texture::InternalFormat::SRGB8_A8)
                                   .build(*r->engine);
        if (!texture) {
            return false;
        }
        // Mips are built on the CPU: Texture::generateMipmaps() stalls the frame readback on
        // the OpenGL backend.
        uint8_t* level = static_cast<uint8_t*>(copyBytes(rgba8, size_t(width) * height * 4));
        if (!level) {
            r->engine->destroy(texture);
            return false;
        }
        uint32_t levelWidth = width;
        uint32_t levelHeight = height;
        for (uint8_t index = 0; index < levels && level; index++) {
            size_t const levelSize = size_t(levelWidth) * levelHeight * 4;
            uint8_t* next = nullptr;
            uint32_t nextWidth = levelWidth > 1 ? levelWidth / 2 : 1;
            uint32_t nextHeight = levelHeight > 1 ? levelHeight / 2 : 1;
            if (index + 1 < levels) {
                next = downsample(level, levelWidth, levelHeight, nextWidth, nextHeight);
            }
            // Filament takes ownership of `level` and frees it once uploaded.
            texture->setImage(*r->engine, index,
                    Texture::PixelBufferDescriptor(level, levelSize, Texture::Format::RGBA,
                            Texture::Type::UBYTE, freeCallback));
            level = next;
            levelWidth = nextWidth;
            levelHeight = nextHeight;
        }
        TextureSampler sampler(TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR,
                TextureSampler::MagFilter::LINEAR, TextureSampler::WrapMode::REPEAT);
        sampler.setAnisotropy(8.0f);
        r->materialInstance->setParameter("baseColorMap", texture, sampler);
        if (r->baseColorMap) {
            r->engine->destroy(r->baseColorMap);
        }
        r->baseColorMap = texture;
        return true;
    } catch (...) {
        return false;
    }
}

// vertices: interleaved [position.xyz, uv.xy, tangent-frame quaternion.xyzw] (9 floats each).
bool sc_renderer_set_mesh(ScRenderer* r, float const* vertices, uint32_t vertex_count,
        uint32_t const* indices, uint32_t index_count, float const* center,
        float const* half_extent) {
    try {
        destroyMesh(r);
        constexpr uint8_t stride = 9 * sizeof(float);
        size_t const vertexBytes = size_t(vertex_count) * stride;
        size_t const indexBytes = size_t(index_count) * sizeof(uint32_t);
        void* vertexCopy = copyBytes(vertices, vertexBytes);
        void* indexCopy = copyBytes(indices, indexBytes);
        if (!vertexCopy || !indexCopy) {
            std::free(vertexCopy);
            std::free(indexCopy);
            return false;
        }

        using Attribute = VertexBuffer::AttributeType;
        r->vertexBuffer = VertexBuffer::Builder()
                                  .vertexCount(vertex_count)
                                  .bufferCount(1)
                                  .attribute(VertexAttribute::POSITION, 0, Attribute::FLOAT3, 0, stride)
                                  .attribute(VertexAttribute::UV0, 0, Attribute::FLOAT2, 12, stride)
                                  .attribute(VertexAttribute::TANGENTS, 0, Attribute::FLOAT4, 20, stride)
                                  .build(*r->engine);
        r->indexBuffer = IndexBuffer::Builder()
                                 .indexCount(index_count)
                                 .bufferType(IndexBuffer::IndexType::UINT)
                                 .build(*r->engine);
        if (!r->vertexBuffer || !r->indexBuffer) {
            std::free(vertexCopy);
            std::free(indexCopy);
            destroyMesh(r);
            return false;
        }
        r->vertexBuffer->setBufferAt(*r->engine, 0,
                VertexBuffer::BufferDescriptor(vertexCopy, vertexBytes, freeCallback));
        r->indexBuffer->setBuffer(*r->engine,
                IndexBuffer::BufferDescriptor(indexCopy, indexBytes, freeCallback));

        r->meshEntity = EntityManager::get().create();
        Box const bounds = {{center[0], center[1], center[2]},
                {half_extent[0], half_extent[1], half_extent[2]}};
        auto const result = RenderableManager::Builder(1)
                                    .boundingBox(bounds)
                                    .material(0, r->materialInstance)
                                    .geometry(0, RenderableManager::PrimitiveType::TRIANGLES,
                                            r->vertexBuffer, r->indexBuffer)
                                    .culling(false)
                                    .receiveShadows(false)
                                    .castShadows(false)
                                    .build(*r->engine, r->meshEntity);
        if (result != RenderableManager::Builder::Success) {
            destroyMesh(r);
            return false;
        }
        r->scene->addEntity(r->meshEntity);
        return true;
    } catch (...) {
        destroyMesh(r);
        return false;
    }
}

void sc_renderer_clear_mesh(ScRenderer* r) {
    try {
        destroyMesh(r);
    } catch (...) {
    }
}

void sc_renderer_set_camera(ScRenderer* r, double left, double right, double bottom, double top,
        double near_plane, double far_plane, double const* eye, double const* target,
        double const* up) {
    try {
        r->camera->setProjection(Camera::Projection::ORTHO, left, right, bottom, top, near_plane,
                far_plane);
        r->camera->lookAt({eye[0], eye[1], eye[2]}, {target[0], target[1], target[2]},
                {up[0], up[1], up[2]});
    } catch (...) {
    }
}

// Renders one frame and schedules an RGBA8 readback (rows top to bottom). Returns false when
// the frame was skipped, in which case `callback` is never invoked and `user` stays owned by
// the caller. On success `callback` runs exactly once from sc_renderer_pump().
bool sc_renderer_render(ScRenderer* r, sc_readback_fn callback, void* user) {
    try {
        if (!r->renderer->beginFrame(r->swapChain)) {
            return false;
        }
        r->renderer->render(r->view);

        size_t const size = size_t(r->width) * size_t(r->height) * 4;
        void* pixels = std::malloc(size);
        auto* readback = new (std::nothrow) Readback{r, callback, user};
        if (!pixels || !readback) {
            std::free(pixels);
            delete readback;
            r->renderer->endFrame();
            return false;
        }
        r->renderer->readPixels(0, 0, r->width, r->height,
                backend::PixelBufferDescriptor(pixels, size, backend::PixelDataFormat::RGBA,
                        backend::PixelDataType::UBYTE, readbackCallback, readback));
        r->pendingReadbacks++;
        r->renderer->endFrame();
        return true;
    } catch (...) {
        return false;
    }
}

// Delivers finished readbacks. The OpenGL backend only retires readPixels() when the driver
// ticks (inside begin/endFrame) or finishes, so while a readback is outstanding and no new
// frame is being submitted, flushAndWait() is needed for it to ever complete.
void sc_renderer_pump(ScRenderer* r) {
    try {
        if (r->pendingReadbacks > 0) {
            r->engine->flushAndWait();
        }
        r->engine->pumpMessageQueues();
    } catch (...) {
    }
}

} // extern "C"
