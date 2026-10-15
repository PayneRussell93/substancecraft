// Narrow C ABI over the Filament C++ API for the SubstanceCraft viewport.
//
// Every entry point catches C++ exceptions so they never unwind into Rust. Buffers handed to
// Filament are copied here and released by Filament's completion callbacks, so callers only
// need to keep their slices alive for the duration of the call.

#include <filament/Camera.h>
#include <filament/ColorGrading.h>
#include <filament/ToneMapper.h>
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

#include <filament-iblprefilter/IBLPrefilterContext.h>

#include <math/mat3.h>

#include <utils/Entity.h>
#include <utils/EntityManager.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace filament;
using utils::Entity;
using utils::EntityManager;

namespace {

constexpr uint32_t MAP_COUNT = 4;
// Emissive maps hold 0..1 sRGB; this many nits make white glow at the default exposure.
constexpr float EMISSIVE_NITS = 40000.0f;
constexpr uint32_t MAX_PRIMITIVES = 256;

struct ScRenderer {
    Engine* engine = nullptr;
    SwapChain* swapChain = nullptr;
    Renderer* renderer = nullptr;
    Scene* scene = nullptr;
    View* view = nullptr;
    Camera* camera = nullptr;
    Entity cameraEntity;
    Entity sunEntity;
    bool sunInScene = true;
    IndirectLight* indirectLight = nullptr;
    // Prefiltered environment behind `indirectLight`, when one is set.
    Texture* reflections = nullptr;
    Texture* irradiance = nullptr;
    IBLPrefilterContext* iblContext = nullptr;
    // Environment rotation about +Y (radians), shared by the light and the background.
    float environmentRotation = 0.0f;
    // Environment background: a full-screen triangle drawn behind the mesh.
    Material* backgroundMaterial = nullptr;
    MaterialInstance* backgroundInstance = nullptr;
    VertexBuffer* backgroundVertices = nullptr;
    IndexBuffer* backgroundIndices = nullptr;
    Entity backgroundEntity;
    bool backgroundWanted = false;
    bool backgroundInScene = false;
    float backgroundBlur = 0.0f;
    // Directional lights of the project's light rig; null until first used.
    std::array<Entity, 4> rigLights{};
    uint32_t rigLightsInScene = 0;
    // View transform and display LUT; null keeps Filament's default grading.
    ColorGrading* colorGrading = nullptr;
    Material* material = nullptr;
    // One instance per mesh primitive (material slot), with its own maps indexed by map slot:
    // 0 base colour, 1 ORM, 2 normal, 3 emissive.
    std::vector<MaterialInstance*> instances;
    std::vector<std::array<Texture*, MAP_COUNT>> maps;
    bool previewMode = false;
    // Shader definition parameters, in ShaderDefinition::parameters order.
    std::array<float, 9> shader{0.5f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 0.0f};
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

// 2x2 box filter. sRGB data is averaged in linear space; linear data (ORM, normals) and alpha
// are averaged directly. Odd edges clamp to the last texel.
uint8_t* downsample(uint8_t const* source, uint32_t width, uint32_t height, uint32_t outWidth,
        uint32_t outHeight, bool srgb) {
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
                if (!srgb) {
                    target[channel] = uint8_t((unsigned(texels[0][channel]) + texels[1][channel] +
                                                      texels[2][channel] + texels[3][channel] + 2) /
                                              4);
                    continue;
                }
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

// Destroys material instances (and their maps) from index `keep` on.
void destroyInstances(ScRenderer* r, size_t keep) {
    while (r->instances.size() > keep) {
        for (Texture* map : r->maps.back()) {
            if (map) r->engine->destroy(map);
        }
        r->engine->destroy(r->instances.back());
        r->instances.pop_back();
        r->maps.pop_back();
    }
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
    destroyInstances(r, 0);
    for (Entity& light : r->rigLights) {
        if (!light.isNull()) {
            engine->destroy(light);
            EntityManager::get().destroy(light);
        }
    }
    if (r->colorGrading) {
        if (r->view) r->view->setColorGrading(nullptr);
        engine->destroy(r->colorGrading);
    }
    if (!r->backgroundEntity.isNull()) {
        engine->destroy(r->backgroundEntity);
        EntityManager::get().destroy(r->backgroundEntity);
    }
    if (r->backgroundInstance) engine->destroy(r->backgroundInstance);
    if (r->backgroundMaterial) engine->destroy(r->backgroundMaterial);
    if (r->backgroundVertices) engine->destroy(r->backgroundVertices);
    if (r->backgroundIndices) engine->destroy(r->backgroundIndices);
    if (r->material) engine->destroy(r->material);
    if (r->indirectLight) engine->destroy(r->indirectLight);
    if (r->reflections) engine->destroy(r->reflections);
    if (r->irradiance) engine->destroy(r->irradiance);
    delete r->iblContext;
    r->iblContext = nullptr;
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

char const* const MAP_PARAMETERS[MAP_COUNT] = {"baseColorMap", "ormMap", "normalMap", "emissiveMap"};

// Base colour and emissive hold colour; ORM and normals are linear data.
bool isSrgbSlot(uint32_t slot) {
    return slot == 0 || slot == 3;
}

// Uploads one material map (RGBA8, rows top to bottom) of material instance `instance` with a
// CPU-built mip chain and binds it to its sampler. Colour maps use sRGB storage and filtering;
// data maps stay linear.
bool uploadMap(ScRenderer* r, uint32_t instance, uint32_t slot, uint32_t width, uint32_t height,
        uint8_t const* rgba8) {
    if (instance >= r->instances.size() || slot >= MAP_COUNT || width == 0 || height == 0) {
        return false;
    }
    bool const srgb = isSrgbSlot(slot);
    uint8_t levels = 1;
    for (uint32_t extent = width > height ? width : height; extent > 1; extent >>= 1) {
        levels++;
    }
    Texture* texture = Texture::Builder()
                               .width(width)
                               .height(height)
                               .levels(levels)
                               .sampler(Texture::Sampler::SAMPLER_2D)
                               .format(srgb ? Texture::InternalFormat::SRGB8_A8
                                            : Texture::InternalFormat::RGBA8)
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
            next = downsample(level, levelWidth, levelHeight, nextWidth, nextHeight, srgb);
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
    r->instances[instance]->setParameter(MAP_PARAMETERS[slot], texture, sampler);
    if (r->maps[instance][slot]) {
        r->engine->destroy(r->maps[instance][slot]);
    }
    r->maps[instance][slot] = texture;
    return true;
}

// Sets the shader definition's parameters on one material instance.
void applyShader(ScRenderer* r, MaterialInstance* instance) {
    auto const& s = r->shader;
    instance->setParameter("specularLevel", s[0]);
    instance->setParameter("clearCoat", s[1]);
    instance->setParameter("clearCoatRoughness", s[2]);
    instance->setParameter("sheenColor", math::float3{s[3], s[4], s[5]});
    instance->setParameter("sheenRoughness", s[6]);
    instance->setParameter("emissiveScale", s[7]);
    instance->setParameter("unlit", s[8]);
}

// Grows or shrinks the instance list to `count`. New instances start with neutral maps: grey
// base colour, AO 1 / roughness 0.58 / metallic 0, a flat normal, and no emission.
bool ensureInstances(ScRenderer* r, uint32_t count) {
    destroyInstances(r, count);
    uint8_t const defaults[MAP_COUNT][4] = {
            {168, 173, 179, 255}, {255, 148, 0, 255}, {128, 128, 255, 255}, {0, 0, 0, 255}};
    while (r->instances.size() < count) {
        MaterialInstance* instance = r->material->createInstance();
        if (!instance) {
            return false;
        }
        instance->setParameter("previewMode", r->previewMode ? 1.0f : 0.0f);
        instance->setParameter("emissiveStrength", EMISSIVE_NITS);
        applyShader(r, instance);
        r->instances.push_back(instance);
        r->maps.push_back({});
        uint32_t const index = uint32_t(r->instances.size() - 1);
        for (uint32_t slot = 0; slot < MAP_COUNT; slot++) {
            if (!uploadMap(r, index, slot, 1, 1, defaults[slot])) {
                return false;
            }
        }
    }
    return true;
}

// A full-screen triangle in device coordinates at the far plane, as Filament draws its skybox.
math::float4 const BACKGROUND_TRIANGLE[3] = {
        {-1.0f, -1.0f, 1.0f, 1.0f}, {3.0f, -1.0f, 1.0f, 1.0f}, {-1.0f, 3.0f, 1.0f, 1.0f}};
uint16_t const BACKGROUND_INDICES[3] = {0, 1, 2};

// Builds the background renderable once its material is loaded.
bool createBackground(ScRenderer* r) {
    if (r->backgroundInstance) {
        return true;
    }
    if (!r->backgroundMaterial) {
        return false;
    }
    r->backgroundVertices = VertexBuffer::Builder()
                                    .vertexCount(3)
                                    .bufferCount(1)
                                    .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT4, 0, 16)
                                    .build(*r->engine);
    r->backgroundIndices = IndexBuffer::Builder()
                                   .indexCount(3)
                                   .bufferType(IndexBuffer::IndexType::USHORT)
                                   .build(*r->engine);
    r->backgroundInstance = r->backgroundMaterial->createInstance();
    if (!r->backgroundVertices || !r->backgroundIndices || !r->backgroundInstance) {
        return false;
    }
    r->backgroundVertices->setBufferAt(*r->engine, 0,
            VertexBuffer::BufferDescriptor(BACKGROUND_TRIANGLE, sizeof(BACKGROUND_TRIANGLE)));
    r->backgroundIndices->setBuffer(*r->engine,
            IndexBuffer::BufferDescriptor(BACKGROUND_INDICES, sizeof(BACKGROUND_INDICES)));
    r->backgroundInstance->setParameter("blur", 0.0f);
    r->backgroundInstance->setParameter("strength", 1.0f);
    r->backgroundEntity = EntityManager::get().create();
    RenderableManager::Builder(1)
            .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, r->backgroundVertices,
                    r->backgroundIndices)
            .material(0, r->backgroundInstance)
            .castShadows(false)
            .receiveShadows(false)
            .priority(7)
            .culling(false)
            .build(*r->engine, r->backgroundEntity);
    return true;
}

// Shows the background when it is wanted and an environment cubemap exists to show.
void syncBackground(ScRenderer* r) {
    bool const show = r->backgroundWanted && r->reflections && createBackground(r);
    // Once built, the instance always holds the current cubemap, shown or not, so no
    // destroyed texture stays bound to it.
    if (r->backgroundInstance && r->reflections) {
        TextureSampler const sampler(TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR,
                TextureSampler::MagFilter::LINEAR, TextureSampler::WrapMode::CLAMP_TO_EDGE);
        r->backgroundInstance->setParameter("environment", r->reflections, sampler);
        float const lastLevel = float(r->reflections->getLevels() - 1);
        r->backgroundInstance->setParameter("blur", r->backgroundBlur * lastLevel);
    }
    if (show && !r->backgroundInScene) {
        r->scene->addEntity(r->backgroundEntity);
        r->backgroundInScene = true;
    } else if (!show && r->backgroundInScene) {
        r->scene->remove(r->backgroundEntity);
        r->backgroundInScene = false;
    }
}

// Aims the background's virtual lens like the camera: view rays turned into world space by
// the camera, then into the environment's frame by undoing its rotation.
void aimBackground(ScRenderer* r) {
    if (!r->backgroundInScene || r->width == 0 || r->height == 0) {
        return;
    }
    math::mat4 const model = r->camera->getModelMatrix();
    math::mat3f const worldFromView(math::float3(model[0].xyz), math::float3(model[1].xyz),
            math::float3(model[2].xyz));
    // Undo the environment turn: a rotation about +Y by the opposite angle.
    math::mat3f const unturn =
            math::mat3f::rotation(-r->environmentRotation, math::float3{0, 1, 0});
    r->backgroundInstance->setParameter("viewToEnvironment", unturn * worldFromView);
    // A 50 degree vertical field of view.
    float const halfHeight = std::tan(0.5f * 50.0f * 3.14159265f / 180.0f);
    float const aspect = float(r->width) / float(r->height);
    r->backgroundInstance->setParameter("lensScale", math::float2{halfHeight * aspect, halfHeight});
}

} // namespace

extern "C" {

ScRenderer* sc_renderer_create(uint8_t const* material, size_t material_len,
        uint8_t const* background, size_t background_len, uint32_t width, uint32_t height) {
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
        // Without its material the background simply stays off.
        r->backgroundMaterial =
                Material::Builder().package(background, background_len).build(*r->engine);
        if (!r->material) {
            destroyAll(r);
            delete r;
            return nullptr;
        }
        // Every sampler needs a texture before the first frame; instances start neutral.
        if (!ensureInstances(r, 1)) {
            destroyAll(r);
            delete r;
            return nullptr;
        }

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

// Upload a map of primitive `primitive`: slot 0 Base Color (sRGB, opacity in alpha), 1 ORM
// (linear), 2 tangent normal (linear), 3 emissive (sRGB).
// Replaces map `slot` of `primitive` with an empty `levels`-level texture; fill it with
// sc_renderer_set_map_region.
bool sc_renderer_create_map(ScRenderer* r, uint32_t primitive, uint32_t slot, uint32_t width,
        uint32_t height, uint32_t levels) {
    try {
        if (primitive >= r->instances.size() || slot >= MAP_COUNT || width == 0 || height == 0 ||
                levels == 0 || levels > 16) {
            return false;
        }
        Texture* texture = Texture::Builder()
                                   .width(width)
                                   .height(height)
                                   .levels(uint8_t(levels))
                                   .sampler(Texture::Sampler::SAMPLER_2D)
                                   .format(isSrgbSlot(slot) ? Texture::InternalFormat::SRGB8_A8
                                                            : Texture::InternalFormat::RGBA8)
                                   .build(*r->engine);
        if (!texture) {
            return false;
        }
        TextureSampler sampler(TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR,
                TextureSampler::MagFilter::LINEAR, TextureSampler::WrapMode::REPEAT);
        sampler.setAnisotropy(8.0f);
        r->instances[primitive]->setParameter(MAP_PARAMETERS[slot], texture, sampler);
        if (r->maps[primitive][slot]) {
            r->engine->destroy(r->maps[primitive][slot]);
        }
        r->maps[primitive][slot] = texture;
        return true;
    } catch (...) {
        return false;
    }
}

// Writes a `width` x `height` block of RGBA8 texels at (`x`, `y`) of mip `level` of a map
// made by sc_renderer_create_map.
bool sc_renderer_set_map_region(ScRenderer* r, uint32_t primitive, uint32_t slot,
        uint32_t level, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
        uint8_t const* rgba8) {
    try {
        if (primitive >= r->instances.size() || slot >= MAP_COUNT || width == 0 || height == 0) {
            return false;
        }
        Texture* texture = r->maps[primitive][slot];
        if (!texture || level >= texture->getLevels() ||
                uint64_t(x) + width > texture->getWidth(level) ||
                uint64_t(y) + height > texture->getHeight(level)) {
            return false;
        }
        size_t const bytes = size_t(width) * height * 4;
        void* copy = copyBytes(rgba8, bytes);
        if (!copy) {
            return false;
        }
        // Filament takes ownership of `copy` and frees it once uploaded.
        texture->setImage(*r->engine, level, x, y, width, height,
                Texture::PixelBufferDescriptor(copy, bytes, Texture::Format::RGBA,
                        Texture::Type::UBYTE, freeCallback));
        return true;
    } catch (...) {
        return false;
    }
}

// Preview mode shows only the base colour map, opaque, on a neutral dielectric.
void sc_renderer_set_preview_mode(ScRenderer* r, bool preview) {
    try {
        r->previewMode = preview;
        for (MaterialInstance* instance : r->instances) {
            instance->setParameter("previewMode", preview ? 1.0f : 0.0f);
        }
    } catch (...) {
    }
}

// vertices: interleaved [position.xyz, uv.xy, tangent-frame quaternion.xyzw] (9 floats each).
// Primitive i draws `primitive_counts[i]` indices from `primitive_offsets[i]` with material
// instance i; each primitive is one mesh material slot.
bool sc_renderer_set_mesh(ScRenderer* r, float const* vertices, uint32_t vertex_count,
        uint32_t const* indices, uint32_t index_count, uint32_t const* primitive_offsets,
        uint32_t const* primitive_counts, uint32_t primitive_count, float const* center,
        float const* half_extent) {
    try {
        destroyMesh(r);
        if (primitive_count == 0 || primitive_count > MAX_PRIMITIVES) {
            return false;
        }
        for (uint32_t i = 0; i < primitive_count; i++) {
            if (primitive_counts[i] == 0 || primitive_offsets[i] > index_count ||
                    primitive_counts[i] > index_count - primitive_offsets[i]) {
                return false;
            }
        }
        if (!ensureInstances(r, primitive_count)) {
            return false;
        }
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
        RenderableManager::Builder builder(primitive_count);
        builder.boundingBox(bounds).culling(false).receiveShadows(false).castShadows(false);
        for (uint32_t i = 0; i < primitive_count; i++) {
            builder.material(i, r->instances[i])
                    .geometry(i, RenderableManager::PrimitiveType::TRIANGLES, r->vertexBuffer,
                            r->indexBuffer, primitive_offsets[i], primitive_counts[i]);
        }
        auto const result = builder.build(*r->engine, r->meshEntity);
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

// Replaces the vertex data of the current mesh (same layout as sc_renderer_set_mesh) without
// touching its indices, primitives, or material instances: a posed copy of the same topology.
bool sc_renderer_set_vertices(ScRenderer* r, float const* vertices, uint32_t vertex_count) {
    try {
        if (!r->vertexBuffer || vertex_count != r->vertexBuffer->getVertexCount()) {
            return false;
        }
        size_t const bytes = size_t(vertex_count) * 9 * sizeof(float);
        void* copy = copyBytes(vertices, bytes);
        if (!copy) {
            return false;
        }
        // Filament takes ownership of `copy` and frees it once uploaded.
        r->vertexBuffer->setBufferAt(*r->engine, 0,
                VertexBuffer::BufferDescriptor(copy, bytes, freeCallback));
        return true;
    } catch (...) {
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
        aimBackground(r);
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

// Lights the scene with an environment: `rgb` is a linear equirectangular image (`width` x
// `height` RGB floats, rows top to bottom, as IBLPrefilterContext reads one), prefiltered here
// into reflection and irradiance cubemaps and applied at `intensity` lux. `rotation` turns the
// environment, and the sun with it, about +Y (radians). With `has_sun`, the directional light
// shines along `sun_direction` (before rotation) at `sun_lux`; otherwise it is removed.
bool sc_renderer_set_environment(ScRenderer* r, float const* rgb, uint32_t width,
        uint32_t height, float intensity, float rotation, bool has_sun,
        float const* sun_direction, float const* sun_color, float sun_lux) {
    Texture* equirect = nullptr;
    Texture* environment = nullptr;
    Texture* reflections = nullptr;
    Texture* irradiance = nullptr;
    try {
        if (width < 2 || height < 1 || width > 8192 || height > 4096 || !(intensity >= 0.0f)) {
            return false;
        }
        if (!r->iblContext) {
            r->iblContext = new IBLPrefilterContext(*r->engine);
        }
        size_t const bytes = size_t(width) * height * 3 * sizeof(float);
        void* copy = copyBytes(rgb, bytes);
        if (!copy) {
            return false;
        }
        // The converter samples the image's mips, which it generates itself, so it needs every
        // level allocated and a format the backend can mipmap.
        uint8_t levels = 1;
        for (uint32_t extent = width; extent > 1; extent >>= 1) {
            levels++;
        }
        equirect = Texture::Builder()
                           .width(width)
                           .height(height)
                           .levels(levels)
                           .sampler(Texture::Sampler::SAMPLER_2D)
                           .format(Texture::InternalFormat::R11F_G11F_B10F)
                           .usage(Texture::Usage::DEFAULT | Texture::Usage::GEN_MIPMAPPABLE)
                           .build(*r->engine);
        if (!equirect) {
            std::free(copy);
            return false;
        }
        // Filament takes ownership of `copy` and frees it once uploaded.
        equirect->setImage(*r->engine, 0,
                Texture::PixelBufferDescriptor(copy, bytes, Texture::Format::RGB,
                        Texture::Type::FLOAT, freeCallback));

        IBLPrefilterContext::EquirectangularToCubemap toCubemap(*r->iblContext);
        environment = toCubemap(equirect);
        IBLPrefilterContext::SpecularFilter specularFilter(*r->iblContext);
        reflections = specularFilter(environment);
        IBLPrefilterContext::IrradianceFilter irradianceFilter(*r->iblContext);
        irradiance = irradianceFilter(environment);
        r->engine->destroy(equirect);
        equirect = nullptr;
        r->engine->destroy(environment);
        environment = nullptr;
        if (!reflections || !irradiance) {
            if (reflections) r->engine->destroy(reflections);
            if (irradiance) r->engine->destroy(irradiance);
            return false;
        }

        math::mat3f const turn = math::mat3f::rotation(rotation, math::float3{0, 1, 0});
        IndirectLight* light = IndirectLight::Builder()
                                       .reflections(reflections)
                                       .irradiance(irradiance)
                                       .intensity(intensity)
                                       .rotation(turn)
                                       .build(*r->engine);
        if (!light) {
            r->engine->destroy(reflections);
            r->engine->destroy(irradiance);
            return false;
        }
        r->scene->setIndirectLight(light);
        IndirectLight* const oldLight = r->indirectLight;
        Texture* const oldReflections = r->reflections;
        Texture* const oldIrradiance = r->irradiance;
        r->indirectLight = light;
        r->reflections = reflections;
        r->irradiance = irradiance;
        r->environmentRotation = rotation;
        // Rebind the background before the old cubemap it may hold goes away.
        syncBackground(r);
        if (oldLight) r->engine->destroy(oldLight);
        if (oldReflections) r->engine->destroy(oldReflections);
        if (oldIrradiance) r->engine->destroy(oldIrradiance);

        // The renderer rotates shading space by the IBL rotation's transpose, so the
        // environment turns by `turn`; the sun turns with it.
        auto& lights = r->engine->getLightManager();
        auto const instance = lights.getInstance(r->sunEntity);
        if (has_sun) {
            math::float3 const direction =
                    turn * math::float3{sun_direction[0], sun_direction[1], sun_direction[2]};
            lights.setDirection(instance, direction);
            lights.setColor(instance, {sun_color[0], sun_color[1], sun_color[2]});
            lights.setIntensity(instance, sun_lux);
            if (!r->sunInScene) {
                r->scene->addEntity(r->sunEntity);
                r->sunInScene = true;
            }
        } else if (r->sunInScene) {
            r->scene->remove(r->sunEntity);
            r->sunInScene = false;
        }
        return true;
    } catch (...) {
        if (equirect) r->engine->destroy(equirect);
        if (environment) r->engine->destroy(environment);
        return false;
    }
}

// Changes the environment's strength and rotation without prefiltering it again.
void sc_renderer_set_environment_transform(ScRenderer* r, float intensity, float rotation,
        bool has_sun, float const* sun_direction) {
    try {
        if (!r->indirectLight) {
            return;
        }
        math::mat3f const turn = math::mat3f::rotation(rotation, math::float3{0, 1, 0});
        r->indirectLight->setIntensity(intensity);
        r->indirectLight->setRotation(turn);
        r->environmentRotation = rotation;
        if (has_sun) {
            auto& lights = r->engine->getLightManager();
            lights.setDirection(lights.getInstance(r->sunEntity),
                    turn * math::float3{sun_direction[0], sun_direction[1], sun_direction[2]});
        }
    } catch (...) {
    }
}

// Sets the viewport shader: nine floats in ShaderDefinition::parameters order, applied to
// every material slot.
void sc_renderer_set_shader(ScRenderer* r, float const* parameters) {
    try {
        for (size_t index = 0; index < r->shader.size(); index++) {
            r->shader[index] = parameters[index];
        }
        for (MaterialInstance* instance : r->instances) {
            applyShader(r, instance);
        }
    } catch (...) {
    }
}

// Shows or hides the environment behind the mesh; `blur` (0..1) moves from the sharp image
// to its roughest prefiltered level.
void sc_renderer_set_background(ScRenderer* r, bool show, float blur) {
    try {
        r->backgroundWanted = show;
        r->backgroundBlur = std::fmin(std::fmax(blur, 0.0f), 1.0f);
        syncBackground(r);
    } catch (...) {
    }
}

// Lights the scene with `count` (at most 4) extra directional lights: `directions` (where each
// shines, xyz), linear `colors` (rgb), and `lux`, three floats per light for the first two.
void sc_renderer_set_rig_lights(ScRenderer* r, uint32_t count, float const* directions,
        float const* colors, float const* lux) {
    try {
        count = count < r->rigLights.size() ? count : uint32_t(r->rigLights.size());
        auto& lights = r->engine->getLightManager();
        for (uint32_t index = 0; index < r->rigLights.size(); index++) {
            Entity& entity = r->rigLights[index];
            if (index < count) {
                math::float3 const direction{directions[index * 3], directions[index * 3 + 1],
                        directions[index * 3 + 2]};
                math::float3 const color{colors[index * 3], colors[index * 3 + 1],
                        colors[index * 3 + 2]};
                if (entity.isNull()) {
                    entity = EntityManager::get().create();
                    LightManager::Builder(LightManager::Type::DIRECTIONAL)
                            .castShadows(false)
                            .build(*r->engine, entity);
                }
                auto const instance = lights.getInstance(entity);
                lights.setDirection(instance, direction);
                lights.setColor(instance, color);
                lights.setIntensity(instance, lux[index]);
                if (index >= r->rigLightsInScene) {
                    r->scene->addEntity(entity);
                }
            } else if (index < r->rigLightsInScene) {
                r->scene->remove(entity);
            }
        }
        r->rigLightsInScene = count;
    } catch (...) {
    }
}

// Sets the view transform: tone mapper 0 standard (Filament's default), 1 ACES, 2 filmic, 3 AgX,
// 4 Khronos PBR Neutral, 5 linear (none); `exposure` in stops; and an optional display LUT of
// `lut_size`³ RGB entries (red fastest) applied to the display-referred sRGB result. Returns
// false when the grading cannot be built.
bool sc_renderer_set_color_grading(ScRenderer* r, uint32_t tone_mapper, float exposure,
        float const* lut, uint32_t lut_size) {
    try {
        ColorGrading* grading = nullptr;
        bool const plain = tone_mapper == 0 && exposure == 0.0f && lut_size == 0;
        if (!plain) {
            LinearToneMapper linear;
            ACESToneMapper aces;
            ACESLegacyToneMapper legacy;
            FilmicToneMapper filmic;
            AgxToneMapper agx;
            PBRNeutralToneMapper neutral;
            ToneMapper const* mapper = &legacy;
            switch (tone_mapper) {
                case 1: mapper = &aces; break;
                case 2: mapper = &filmic; break;
                case 3: mapper = &agx; break;
                case 4: mapper = &neutral; break;
                case 5: mapper = &linear; break;
                default: break;
            }
            ColorGrading::Builder builder;
            builder.quality(ColorGrading::QualityLevel::HIGH).toneMapper(mapper).exposure(exposure);
            if (lut && lut_size >= 2 && lut_size <= 255) {
                size_t const entries = size_t(lut_size) * lut_size * lut_size;
                utils::FixedCapacityVector<math::float3> data(entries);
                for (size_t index = 0; index < entries; index++) {
                    data[index] = {lut[index * 3], lut[index * 3 + 1], lut[index * 3 + 2]};
                }
                builder.customLut(std::move(data), uint8_t(lut_size));
            }
            grading = builder.build(*r->engine);
            if (!grading) {
                return false;
            }
        }
        r->view->setColorGrading(grading);
        if (r->colorGrading) {
            r->engine->destroy(r->colorGrading);
        }
        r->colorGrading = grading;
        return true;
    } catch (...) {
        return false;
    }
}

// Screen-space global illumination: ambient occlusion with bent normals and contact shadows
// from the sun, plus screen-space reflections, over `radius` world units (the mesh's scale).
void sc_renderer_set_global_illumination(ScRenderer* r, bool enabled, float radius) {
    try {
        float const reach = std::fmax(radius, 1e-4f);
        View::AmbientOcclusionOptions ao;
        ao.enabled = enabled;
        ao.radius = reach * 0.3f;
        ao.bias = reach * 0.0005f;
        ao.intensity = 1.0f;
        ao.power = 1.2f;
        ao.bentNormals = true;
        ao.quality = View::QualityLevel::HIGH;
        ao.lowPassFilter = View::QualityLevel::HIGH;
        ao.upsampling = View::QualityLevel::HIGH;
        ao.ssct.enabled = enabled && r->sunInScene;
        ao.ssct.shadowDistance = reach * 0.3f;
        ao.ssct.contactDistanceMax = reach;
        if (r->sunInScene) {
            auto& lights = r->engine->getLightManager();
            ao.ssct.lightDirection = lights.getDirection(lights.getInstance(r->sunEntity));
        }
        r->view->setAmbientOcclusionOptions(ao);
        View::ScreenSpaceReflectionsOptions ssr;
        ssr.enabled = enabled;
        ssr.thickness = reach * 0.05f;
        ssr.bias = reach * 0.005f;
        ssr.maxDistance = reach * 2.0f;
        r->view->setScreenSpaceReflectionsOptions(ssr);
    } catch (...) {
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
