// Some code borrowed from:
//
// compute shaders tutorial
// Dr Anton Gerdelan <gerdela@scss.tcd.ie>
// Trinity College Dublin, Ireland
// 26 Feb 2016


#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <map>
#include <cstdio>
#include <sstream>
#include <ostream>
#include <fstream>
#include <iostream>
#include "imgui/imgui.h"
#include "imgui/examples/imgui_impl_sdl.h"
#include "imgui/examples/imgui_impl_opengl3.h"

#include "gl_renderer.hpp"
#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "shader/input.hpp"

using std::string;
using std::endl;

extern int x_batch;
static float tone_exposure = 0.25;
static float tone_gamma = 2.2;

constexpr int OPENGL_MAJOR_VERSION = 4;
constexpr int OPENGL_MINOR_VERSION = 5;
constexpr SDL_GLprofile OPENGL_PROFILE = SDL_GLprofile::SDL_GL_CONTEXT_PROFILE_CORE;

std::vector<std::string> split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(std::move(item));
  }
  return elems;
}

std::map<int, std::pair<int, std::string>> line_map;



void print_shader_info_log( GLuint shader ) {
  int max_length = 4096;
  int actual_length = 0;
  char slog[4096];
  glGetShaderInfoLog( shader, max_length, &actual_length, slog );
  fprintf( stderr, "shader info log for GL index %u\n%s\n", shader, slog);
}

void print_program_info_log( GLuint program ) {
  int max_length = 4096;
  int actual_length = 0;
  char plog[4096];
  glGetProgramInfoLog( program, max_length, &actual_length, plog );
  fprintf( stderr, "program info log for GL index %u\n%s\n", program, plog );
}

bool check_shader_errors( GLuint shader ) {
  GLint params = -1;
  glGetShaderiv( shader, GL_COMPILE_STATUS, &params );
  if ( GL_TRUE != params ) {
    fprintf( stderr, "ERROR: shader %u did not compile\n", shader );
    print_shader_info_log( shader );
    return false;
  }
  return true;
}

bool check_program_errors( GLuint program ) {
  GLint params = -1;
  glGetProgramiv( program, GL_LINK_STATUS, &params );
  if ( GL_TRUE != params ) {
    fprintf( stderr, "ERROR: program %u did not link\n", program );
    print_program_info_log( program );
    return false;
  }
  return true;
}

GLuint create_quad_vao() {
  GLuint vao = 0, vbo = 0;
  float verts[] = { -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData( GL_ARRAY_BUFFER, 16 * sizeof( float ), verts, GL_STATIC_DRAW );
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );
  glEnableVertexAttribArray( 0 );
  GLintptr stride = 4 * sizeof( float );
  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, stride, NULL );
  glEnableVertexAttribArray( 1 );
  GLintptr offset = 2 * sizeof( float );
  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, (GLvoid *)offset );
  return vao;
}

// this is the quad's vertex shader in an ugly C string
const char *vert_shader_str =
R"(
#version 430
layout (location = 0) in vec2 vp;
layout (location = 1) in vec2 vt;
out vec2 st;

void main () {
  st = vt;
  gl_Position = vec4 (vp, 0.0, 1.0);
}
)";

// this is the quad's fragment shader in an ugly C string
const char *frag_shader_str =
R"(#version 430
in vec2 st;
uniform sampler2D img;
uniform float mul;
out vec4 fc;

void main () {
  vec4 c = sqrt(texture (img, st) * mul);
  // FIXME
  if (c.x < 0) c.x = 0;
  if (c.y < 0) c.y = 0;
  if (c.z < 0) c.z = 0;
  float m =  max(c.x, max(c.y, c.z));
  if (m < 1) {
    fc = c;
    return;
  }
  float total = c.x + c.y + c.z;
  if (total > 3) {
    fc = vec4(1,1,1,1);
    return;
  }
  float scale = (3 - total) / (3 * m - total);
  float grey = 1 - scale * m;
  fc = vec4(grey + scale * c.x,
                grey + scale * c.y,
                grey + scale * c.z, 1);
}
)";

GLuint create_quad_program() {
  GLuint program = glCreateProgram();
  GLuint vert_shader = glCreateShader( GL_VERTEX_SHADER );
  glShaderSource( vert_shader, 1, &vert_shader_str, NULL );
  glCompileShader( vert_shader );
  check_shader_errors( vert_shader );
  glAttachShader( program, vert_shader );
  GLuint frag_shader = glCreateShader( GL_FRAGMENT_SHADER );
  glShaderSource( frag_shader, 1, &frag_shader_str, NULL );
  glCompileShader( frag_shader );
  check_shader_errors( frag_shader );
  glAttachShader( program, frag_shader );
  glLinkProgram( program );
  check_program_errors( program );
  return program;
}

OpenglRenderer::~OpenglRenderer() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();
  if (context_ != nullptr) {
    SDL_GL_DeleteContext(context_);
  }
  if (window_ != nullptr) {
    SDL_DestroyWindow(window_);
  }
}

#include "shader/input.hpp"

enum InputType {
  IN_VEC3,
  IN_FLOAT,
  IN_INT,
};

std::map<std::string, InputType> inputTypeMap = {
  {"vec3", IN_VEC3 },
  {"float", IN_FLOAT },
  {"int", IN_INT },
};

InputType LookupInputType(const char* t) {
  auto it = inputTypeMap.find(t);
  if (it == inputTypeMap.end()) {
    std::cerr << "Unknown input type for model: " << t << endl;
    exit(1);
  }
  return it->second;
}

struct ModelInput {
  InputType type;
  const char* name;
  void* value_ptr;
} model_inputs[] {
#define INPUT(type, name, value) { LookupInputType(#type), #name, &name },
#include "shader/input.hpp"
#undef INPUT
};

#undef uniform
#define uniform extern
#define NO_INIT
#include "shader/struct_input.hpp"
#undef uniform

void OpenglRenderer::bindTexture(int idx, Texture& tex) {
  GLuint ssbo;
  glGenBuffers(1, &ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
  const auto& t = tex.Export();
  glBufferData(GL_SHADER_STORAGE_BUFFER,
      t.size(), &t[0], GL_STATIC_COPY);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, idx, ssbo);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind
}

#define RT_CHECK_ERROR_NO_CONTEXT( func ) \
  do { \
    RTresult code = func; \
    if (code != RT_SUCCESS) \
      std::cerr << "ERROR: Function " << #func << std::endl; \
  } while (0)

namespace {

void readTextureLayer(
    optix::Context ctx, 
    const string& layer,
    int type,
    int width,
    int height,
    const std::vector<unsigned char> bytes) {
  auto tex = ctx->createTextureSampler();
  tex->setWrapMode(0, RT_WRAP_REPEAT);
  tex->setWrapMode(1, RT_WRAP_REPEAT);
  tex->setWrapMode(2, RT_WRAP_REPEAT);
  tex->setFilteringModes(
      RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
  tex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);

  // FIXME: use Normalized for normals texture
//  tex->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
  optix::Buffer buffer;
  if (type == 0) {
    buffer = ctx->createBuffer(
        RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, 1024, 1024);
    unsigned char *dst = (unsigned char*) buffer->map(
        0, RT_BUFFER_MAP_WRITE_DISCARD);
    for (size_t i = 0; i < bytes.size(); i+=3) {
      *dst++ = bytes[i];
      *dst++ = bytes[i+1];
      *dst++ = bytes[i+2];
      *dst++ = 255;
    }
  } else if (type == 1) {
    buffer = ctx->createBuffer(
        RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1024, 1024);
    float *dst = (float*) buffer->map(
        0, RT_BUFFER_MAP_WRITE_DISCARD);
    for (size_t i = 0; i < bytes.size(); i+=3) {
      vec3 n = normalize(vec3{
        (float(bytes[i]) - 128.f) / 256.f,
        (float(bytes[i+1]) - 128.f) / 256.f,
        (float(bytes[i+2]) - 128.f) / 256.f});

      *dst++ = n.x;
      *dst++ = n.y;
      *dst++ = n.z;
      *dst++ = 0;
    }
  } else {
    buffer = ctx->createBuffer(
        RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 1024, 1024);
    unsigned char *dst = (unsigned char*) buffer->map(
        0, RT_BUFFER_MAP_WRITE_DISCARD);
    memcpy(dst, &bytes[0], bytes.size());
  }
  buffer->unmap(0);
  tex->setBuffer(buffer);

  ctx[layer]->setTextureSampler(tex);
}

void loadTexture(optix::Context ctx, const string& name, const Texture& tex) {
  readTextureLayer(ctx, name + "_albedo", 0, tex.width, tex.height, tex.albedo);
  readTextureLayer(ctx, name + "_normals", 1, tex.width, tex.height, tex.normals);
  readTextureLayer(ctx, name + "_roughness", 2, tex.width, tex.height, tex.roughness);
  ctx[name + "_specular_exponent"]->setFloat(tex.specular_exponent_); 
  ctx[name + "_diffuse_ammount"]->setFloat(tex.diffuse_ammount_); 
}

}  // namespace

static float denoise_blend = 0;

void OpenglRenderer::initRenderer() {
  ctx_ = optix::Context::create();
  std::vector<int> devices = {0};
  ctx_->setDevices(devices.begin(), devices.end());
  devices = ctx_->getEnabledDevices();
  if (devices.size() == 0) {
    std::cerr << "No Optix devices\n";
    exit(1);
  }
  std::cout << "Using " << ctx_->getDeviceName(devices[0]) << std::endl;
  auto ray_prog = ctx_->createProgramFromPTXFile("ray.ptx", "ray");
  auto exc_prog = ctx_->createProgramFromPTXFile("exception.ptx", "exception");

  ctx_->setEntryPointCount(1); // 0 = render
  ctx_->setRayTypeCount(0);    // This initial demo is not shooting any rays.
  ctx_->setStackSize(1024);

#if USE_DEBUG_EXCEPTIONS
  // Disable this by default for performance, otherwise the stitched PTX code will have lots of exception handling inside. 
  ctx_->setPrintEnabled(true);
  //ctx_->setPrintLaunchIndex(256, 256);
  ctx_->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#endif 

  // Add ctx_-global variables here.
  ctx_["sysColorBackground"]->setFloat(0.462745f, 0.72549f, 0.0f);

  // This demo just writes into the output buffer, so use RT_BUFFER_OUTPUT as type.
  // In case of an OpenGL interop buffer, that is automatically registered with CUDA now! Must unregister/register around size changes.
  bufferResult_ = ctx_->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width_, height_);
  ctx_["sysOutputBuffer"]->set(bufferResult_);
  bufferAlbedo_ = ctx_->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width_, height_);
  ctx_["sysAlbedoBuffer"]->set(bufferAlbedo_);
  bufferNormal_ = ctx_->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width_, height_);
  ctx_["sysOutNormalBuffer"]->set(bufferNormal_);

  ctx_->setRayGenerationProgram(0, ray_prog);
  ctx_->setExceptionProgram(0, exc_prog);

  loadTexture(ctx_, "wall", *wall_tex);
  loadTexture(ctx_, "ceiling", *ceiling_tex);
  loadTexture(ctx_, "floor", *floor_tex);

//  auto geo = ctx_->createGeometry();
//  auto group = ctx_->createGeometryGroup();
  ctx_["sysBatchSize"]->setUint(x_batch);


  auto anyhit = ctx_->createProgramFromPTXFile("anyhit.ptx", "anyhit");

  auto tonemapStage = ctx_->createBuiltinPostProcessingStage("TonemapperSimple");
  auto denoiserStage = ctx_->createBuiltinPostProcessingStage("DLDenoiser");
//        if (trainingDataBuffer)
//        {
//            Variable trainingBuff = denoiserStage->declareVariable("training_data_buffer");
//            trainingBuff->set(trainingDataBuffer);
//        }

  bufferTonemap_ = ctx_->createBuffer(RT_BUFFER_INPUT_OUTPUT,
      RT_FORMAT_FLOAT4, width_, height_);
  ctx_["tone_buf"]->setBuffer(bufferTonemap_);
  bufferOutput_ = ctx_->createBuffer(
      RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width_, height_);

  tonemapStage->declareVariable("input_buffer")->set(bufferResult_);
  tonemapStage->declareVariable("output_buffer")->set(bufferTonemap_);
  exposure_ = tonemapStage->declareVariable("exposure");
  exposure_->setFloat(tone_exposure);
  gamma_ = tonemapStage->declareVariable("gamma");
  gamma_->setFloat(tone_gamma);


  denoiserStage->declareVariable("input_buffer")->set(bufferTonemap_);
  denoiserStage->declareVariable("output_buffer")->set(bufferOutput_);
  denoiseBlend_ = denoiserStage->declareVariable("blend");
  denoiseBlend_->setFloat(denoise_blend);
  denoiserStage->declareVariable("input_albedo_buffer")->set(bufferAlbedo_);
//  denoiserStage->declareVariable("input_normal_buffer")->set(bufferNormal_);

  commandListTonemap_ = ctx_->createCommandList();
  commandListTonemap_->appendLaunch(0, width_, height_);
  commandListTonemap_->appendPostprocessingStage(tonemapStage, width_, height_);
  commandListTonemap_->finalize();

  commandListDenoiser_ = ctx_->createCommandList();
  commandListDenoiser_->appendLaunch(0, width_, height_);
  commandListDenoiser_->appendPostprocessingStage(tonemapStage, width_, height_);
  commandListDenoiser_->appendPostprocessingStage(denoiserStage, width_, height_);
  commandListDenoiser_->finalize();
}

bool OpenglRenderer::setup() {
  initRenderer();

  // set up shaders and geometry for full-screen quad
  // moved code to gl_utils.cpp
  quad_vao = create_quad_vao();
  quad_program = create_quad_program();

  // texture handle and dimensions
  tex_output = 0;
  { // create the texture
    glGenTextures( 1, &tex_output );
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, tex_output );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT,
        NULL );
  }

  glUseProgram( quad_program );
  post_processor_mul_ = glGetUniformLocation(quad_program, "mul");
  frame_num = 0;


  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  io_ = &ImGui::GetIO();
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls

  ImGui_ImplSDL2_InitForOpenGL(window_, context_);
  ImGui_ImplOpenGL3_Init("#version 150");

  // Setup style
  ImGui::StyleColorsDark();

  return true;
}

std::unique_ptr<Renderer> OpenglRenderer::Create(int window_width, int window_height) {
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, OPENGL_PROFILE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, OPENGL_MAJOR_VERSION);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, OPENGL_MINOR_VERSION);

  std::unique_ptr<OpenglRenderer> r(new OpenglRenderer());

  r->width_ = window_width;
  r->height_ = window_height;

  r->window_ = SDL_CreateWindow(
      "glray",
      SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
      r->width_, r->height_,
      SDL_WINDOW_OPENGL);

  if (r->window_ == nullptr) {
    std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
    return nullptr;
  }

  r->context_ = SDL_GL_CreateContext(r->window_);

  if (r->context_ == nullptr) {
    std::cerr << "Failed to initialize OpenGL 4.5: " << SDL_GetError() << std::endl;
    return nullptr;
  }

  glewExperimental = GL_TRUE;
  glewInit();

  const unsigned char *version = glGetString(GL_VERSION);
  if (version == nullptr) {
    std::cerr << "Problem with OpenGL:" << SDL_GetError() << std::endl;
    return nullptr;
  }
  std::cerr << "OpenGL version: " << version << endl;

  int res = SDL_GL_MakeCurrent(r->window_, r->context_);
  if (res != 0) {
    std::cerr << "Problem with OpenGL:" << SDL_GetError() << std::endl;
    return nullptr;
  }

  if (!r->setup()) {
    return nullptr;
  }
  std::cerr << "Created OpenGL renderer\n";
  return r;
}

void OpenglRenderer::reset_accumulate() {
  frame_num = 0;
}

void OpenglRenderer::ProcessEvent(SDL_Event* event) {
  ImGui_ImplSDL2_ProcessEvent(event);
}
bool OpenglRenderer::WantCaptureMouse() {
  return io_ != nullptr && io_->WantCaptureMouse;
}

bool OpenglRenderer::WantCaptureKeyboard() {
  return io_ != nullptr && io_->WantCaptureKeyboard;
}

bool show_demo_window = false;
bool show_settings = true;
vec3 clear_color;
vec3 absorption_color = vec3(0.17, 0.17, 0.53);
float absorption_intensity = 1.0f;
bool no_light_rays = false;
int output_selector = 3;

void OpenglRenderer::draw() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL2_NewFrame(window_);
  ImGui::NewFrame();
  bool a = requireInit;

  // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
  if (show_demo_window) {
    ImGui::ShowDemoWindow(&show_demo_window);
  }

  if (show_settings) {
    ImGui::Begin("Settings");

    ImGui::Checkbox("Demo Window", &show_demo_window);

    ImGui::DragFloat("Brightness", &brightness, 0.05, 0.01f, 100.0f);
    if (ImGui::SliderFloat("Exposure", &tone_exposure, 0.0f, 1.0f)) {
      exposure_->setFloat(tone_exposure);
    }
    if (ImGui::SliderFloat("Gamma", &tone_gamma, 0.0f, 5.0f)) {
      gamma_->setFloat(tone_gamma);
    }
    if (ImGui::SliderFloat("Denoise", &denoise_blend, 0.0f, 1.0f)) {
      denoiseBlend_->setFloat(denoise_blend);
    }
    ImGui::SliderInt("Output", &output_selector, 0, 4);

    a |= ImGui::Checkbox("No light rays", &no_light_rays);
    a |= ImGui::DragFloat("Lense Size", &lense_blur, 0.0001, 0.0001f, 0.1f, "%0.4f");
    a |= ImGui::DragFloat("Focus Distance", &focused_distance, 0.05, 0.01f, 10.0f);
    a |= ImGui::DragFloat("Light Size", &light_size, 0.01, 0.01f, 4.0f);
    a |= ImGui::SliderInt("Max Depth", &max_depth, 1, 20);
    a |= ImGui::ColorEdit3("Absorption Color", (float*)&absorption_color, 0);
    a |= ImGui::DragFloat("Absorption Intensitiy", &absorption_intensity, 0.05, 0.05, 10);
    a |= ImGui::DragFloat("Refraction Index", &glass_refraction_index, 0.01, 0.9, 5);
    a |= ImGui::SliderInt("Max Internal Reflections", &max_internal_reflections, 0, 30);
    a |= ImGui::DragFloat3("Room Min", (float*)&room.a_, 0.1, -50, 50);
    a |= ImGui::DragFloat3("Room Max", (float*)&room.b_, 0.1, -50, 50);

    if (ImGui::TreeNode("Balls")) {
      for (size_t i = 0; i < LENGTH(balls); i++)
        if (ImGui::TreeNode((void*)(intptr_t)i, "Ball %ld", i)) {
          a |= ImGui::DragFloat3("Position", (float*)&balls[i].position_, 0.01, -10, 10);
          a |= ImGui::DragFloat("Size", (float*)&balls[i].size_, 0.1, 0, 5);
          a |= ImGui::ColorEdit3("Color", (float*)&balls[i].color_);
          a |= ImGui::DragFloat("Diffuse ammount", (float*)&balls[i].material_.diffuse_ammount_, 0.01, 0, 1);
          a |= ImGui::DragFloat("Specular", (float*)&balls[i].material_.specular_exponent_, 0.1, 1, 100000, "%f", 2);
          ImGui::TreePop();
        }
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Materials")) {
      if (ImGui::TreeNode("Wall")) {
        a |= ImGui::SliderFloat("Diffuse ammount", &wall_tex->diffuse_ammount_, 0, 1);
        a |= ImGui::SliderFloat("Specular", &wall_tex->specular_exponent_, 1, 100000, "%f", 2);
        ImGui::TreePop();
      }
      if (ImGui::TreeNode("Ceiling")) {
        a |= ImGui::SliderFloat("Diffuse ammount", &ceiling_tex->diffuse_ammount_, 0, 1);
        a |= ImGui::SliderFloat("Specular", &ceiling_tex->specular_exponent_, 1, 100000, "%f", 2);
        ImGui::TreePop();
      }

      if (ImGui::TreeNode("Floor")) {
        a |= ImGui::SliderFloat("Diffuse ammount", &floor_tex->diffuse_ammount_, 0, 1);
        a |= ImGui::SliderFloat("Specular", &floor_tex->specular_exponent_, 1, 100000, "%f", 2);
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }

    ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
  }
  if (a) {
    reset_accumulate();
    absorption = absorption_color * absorption_intensity;
    light_size2 = light_size * light_size;
    ctx_["sysLenseBlur"]->setFloat(lense_blur);
    ctx_["sysFocusedDistance"]->setFloat(focused_distance);
    ctx_["sysLightSize"]->setFloat(light_size);
    ctx_["sysLightSize2"]->setFloat(light_size2);
    ctx_["sysMaxDepth"]->setUint(max_depth);
    ctx_["sysRefractionIndex"]->setFloat(glass_refraction_index);
    ctx_["sysAbsorption"]->set3fv((float*)&absorption);
    ctx_["room"]->setUserData(sizeof(room), &room);
    ctx_["sysMaxInternalReflections"]->setUint(max_internal_reflections);
    ctx_["sysTracerFlags"]->setUint(no_light_rays ? 2 : 0); 

    for (size_t i = 0; i < LENGTH(balls); i++) {
      balls[i].size2_ = balls[i].size_ * balls[i].size_;
      balls[i].inv_size_ = 1.f / balls[i].size_;
      if (i == 0) {
        bbox.a_ = balls[i].position_ - vec3(balls[i].size_);
        bbox.b_ = balls[i].position_ + vec3(balls[i].size_);
      } else {
        bbox.a_ = min(bbox.a_, balls[i].position_ - vec3(balls[i].size_));
        bbox.b_ = max(bbox.b_, balls[i].position_ + vec3(balls[i].size_));
      }
    }

    ctx_["balls"]->setUserData(sizeof(balls), &balls);
    ctx_["bbox"]->setUserData(sizeof(bbox), &bbox);
    ctx_["wall_specular_exponent"]->setFloat(wall_tex->specular_exponent_); 
    ctx_["wall_diffuse_ammount"]->setFloat(wall_tex->diffuse_ammount_); 
    ctx_["ceiling_specular_exponent"]->setFloat(ceiling_tex->specular_exponent_); 
    ctx_["ceiling_diffuse_ammount"]->setFloat(ceiling_tex->diffuse_ammount_); 
    ctx_["floor_specular_exponent"]->setFloat(floor_tex->specular_exponent_); 
    ctx_["floor_diffuse_ammount"]->setFloat(floor_tex->diffuse_ammount_); 
    requireInit = false;
  }

  ctx_["sysFrameNum"]->setUint(frame_num);
  ctx_["sysMaxRays"]->setUint(max_rays);
  ctx_["sysSight"]->set3fv((float*)&sight);
  ctx_["sysSightX"]->set3fv((float*)&sight_x);
  ctx_["sysSightY"]->set3fv((float*)&sight_y);
  ctx_["sysViewer"]->set3fv((float*)&viewer);

  ctx_->launch(0, width_ / x_batch, height_);
  switch (output_selector) {
    case 2: commandListTonemap_->execute(); break;
    case 3: commandListDenoiser_->execute(); break;
    default: break;
  }
  ImGui::Render();

  glActiveTexture( GL_TEXTURE0 );
  glBindTexture( GL_TEXTURE_2D, tex_output );
  optix::Buffer selected;
  switch (output_selector) {
    case 0: selected = bufferAlbedo_; break;
    case 1: selected = bufferResult_; break;
    case 2: selected = bufferTonemap_; break;
    case 3: selected = bufferOutput_; break;
    case 4: selected = bufferNormal_; break;
    default: selected = bufferOutput_; break;
  }
  const void* data = selected->map(0, RT_BUFFER_MAP_READ);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) width_, (GLsizei) height_, 0, GL_RGBA, GL_FLOAT, data); // RGBA32F
  selected->unmap();

  // prevent sampling befor all writes to image are done
  glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

//  glClear( GL_COLOR_BUFFER_BIT );
  glUseProgram( quad_program );
  frame_num += max_rays;
//  glUniform1f(post_processor_mul_, 1.f/frame_num * brightness);
  glUniform1f(post_processor_mul_, output_selector == 0 || output_selector == 4? 1 : brightness);
  glBindVertexArray( quad_vao );
  glActiveTexture( GL_TEXTURE0 );
  glBindTexture( GL_TEXTURE_2D, tex_output );
  glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  SDL_GL_SwapWindow(window_);

  if ((frame_num & (frame_num - 1)) == 0 && frame_num > 8) {
    printf("Frame num: %d\n", frame_num);
    fflush(stdout);
  }
}

