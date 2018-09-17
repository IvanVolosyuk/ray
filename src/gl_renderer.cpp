// Some code borrowed from:
//
// compute shaders tutorial
// Dr Anton Gerdelan <gerdela@scss.tcd.ie>
// Trinity College Dublin, Ireland
// 26 Feb 2016

#include "gl_renderer.hpp"

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

using std::string;
using std::endl;

constexpr int OPENGL_MAJOR_VERSION = 4;
constexpr int OPENGL_MINOR_VERSION = 5;
constexpr SDL_GLprofile OPENGL_PROFILE = SDL_GLprofile::SDL_GL_CONTEXT_PROFILE_CORE;

struct shader_data_t
{
  float viewer[4];
  float sight[4];
} shader_data;

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

void make_shader_map(const string& shader) {
  line_map.clear();
  auto lines = split(shader, '\n');
  std::regex marker(R"regexp(// GENERATED DONT EDIT (.*) "(.*)".*)regexp");
  for (size_t i = 0; i < lines.size(); i++) {
    const string& line = lines[i];
    std::smatch m;
    if (std::regex_match (line, m, marker)) {
      int linenum = std::stoi(m[1].str());
      string filename = m[2].str();
//      fprintf(stderr, "Added: %d %d %s\n", i, linenum, filename.c_str());
      line_map[i] = std::make_pair(linenum, filename);
    }
  }
}

string reformat_errors(const string& input) {
  std::stringstream out;
  auto linenum_mesa = std::regex(R"([0-9]+:([0-9]+)\([0-9]+\)(: .*))");
  auto linenum_nvidia = std::regex(R"(.*[0-9]+\(([0-9]+)\)( *: .*))");


  for (const auto& line : split(input, '\n')) {
    std::smatch m;
    auto fix = [&]() {
      int err_line = std::stoi(m[1].str()) - 1;
      auto lookup = --line_map.upper_bound(err_line);
      if (lookup != line_map.begin()) {
        int offset = err_line - lookup->first;
        int source_line = lookup->second.first + offset - 1;
        string error_msg = m[2].str() ;
        out << lookup->second.second
            << ":" << source_line
            << " s" << err_line << error_msg.c_str() << endl;
      } else {
        out << "Unmatched: " << err_line << " >> " << line << endl;
      }
    };

    if (std::regex_match (line, m, linenum_mesa)) {
      fix();
    } else if(std::regex_match (line, m, linenum_nvidia)) {
      fix();
    } else {
      out << line << endl;
    }
  }
  return out.str();
}

void print_shader_info_log( GLuint shader ) {
  int max_length = 4096;
  int actual_length = 0;
  char slog[4096];
  glGetShaderInfoLog( shader, max_length, &actual_length, slog );
  string reformated = reformat_errors(slog);
  fprintf( stderr, "shader info log for GL index %u\n%s\n", shader, reformated.c_str());
}

void print_program_info_log( GLuint program ) {
  int max_length = 4096;
  int actual_length = 0;
  char plog[4096];
  glGetProgramInfoLog( program, max_length, &actual_length, plog );
  string reformated = reformat_errors(plog);
  fprintf( stderr, "program info log for GL index %u\n%s\n", program, reformated.c_str() );
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
  check_shader_errors( vert_shader ); // code moved to gl_utils.cpp
  glAttachShader( program, vert_shader );
  GLuint frag_shader = glCreateShader( GL_FRAGMENT_SHADER );
  glShaderSource( frag_shader, 1, &frag_shader_str, NULL );
  glCompileShader( frag_shader );
  check_shader_errors( frag_shader ); // code moved to gl_utils.cpp
  glAttachShader( program, frag_shader );
  glLinkProgram( program );
  check_program_errors( program ); // code moved to gl_utils.cpp
  return program;
}

OpenglRenderer::~OpenglRenderer() {
  if (context_ != nullptr) {
    SDL_GL_DeleteContext(context_);
  }
  if (window_ != nullptr) {
    SDL_DestroyWindow(window_);
  }
}

#include "shader/input.h"

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
#include "shader/input.h"
#undef INPUT
};

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

bool OpenglRenderer::setup() {
  std::ifstream t("shader.comp");
  std::string compute_shader((std::istreambuf_iterator<char>(t)),
      std::istreambuf_iterator<char>());
  if (compute_shader.empty()) {
    std::cerr << "Failed to load 'shader.comp'\n";
    return false;
  }
  make_shader_map(compute_shader);
  const char *cs = compute_shader.c_str();

  // set up shaders and geometry for full-screen quad
  // moved code to gl_utils.cpp
  quad_vao = create_quad_vao();
  quad_program = create_quad_program();

  ray_program = 0;
  { // create the compute shader
    GLuint ray_shader = glCreateShader( GL_COMPUTE_SHADER );
    glShaderSource( ray_shader, 1, &cs, NULL );
    glCompileShader( ray_shader );
    ( check_shader_errors( ray_shader ) ); // code moved to gl_utils.cpp
    ray_program = glCreateProgram();
    glAttachShader( ray_program, ray_shader );
    glLinkProgram( ray_program );
    ( check_program_errors( ray_program ) ); // code moved to gl_utils.cpp
  }

  // texture handle and dimensions
  tex_output = 0;
  { // create the texture
    glGenTextures( 1, &tex_output );
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, tex_output );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    // linear allows us to scale the window up retaining reasonable quality
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    // same internal format as compute shader input
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT,
        NULL );
    // bind to image unit so can write to specific pixels from the shader
    glBindImageTexture( 0, tex_output, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F );
  }

  { // query up the workgroups
    int work_grp_size[3], work_grp_inv;
    // maximum global work group (total work in a dispatch)
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_size[0] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_size[1] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_size[2] );
    printf( "max global (total) work group size x:%i y:%i z:%i\n", work_grp_size[0],
        work_grp_size[1], work_grp_size[2] );
    // maximum local work group (one shader's slice)
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1] );
    glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2] );
    printf( "max local (in one shader) work group sizes x:%i y:%i z:%i\n",
        work_grp_size[0], work_grp_size[1], work_grp_size[2] );
    // maximum compute shader invocations (x * y * z)
    glGetIntegerv( GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv );
    printf( "max computer shader invocations %i\n", work_grp_inv );
  }

  // Init inputs
  glUseProgram( ray_program );
  for (size_t i = 0; i < LENGTH(model_inputs); i++) {
    GLint location = glGetUniformLocation(ray_program, model_inputs[i].name);
    if (location == -1) {
      std::cerr << "Location " << model_inputs[i].name
        << " undefined: " << SDL_GetError() << endl;
      return false;
    }
    inputs_.push_back(location);
  }
  bindTexture(0, *floor_tex);
  bindTexture(1, *wall_tex);
  bindTexture(2, *ceiling_tex);

  glUseProgram( quad_program );
  post_processor_mul_ = glGetUniformLocation(quad_program, "mul");
  frame_num = 0;

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
      1920, 1080,
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

extern int x_batch;

void OpenglRenderer::draw() {

  glUseProgram( ray_program );
  for (size_t i = 0; i < LENGTH(model_inputs); i++) {
    const ModelInput& input = model_inputs[i];
    switch(input.type) {
      case IN_VEC3:
        {
          const vec3* value = (vec3*) input.value_ptr;
          glUniform3f(inputs_[i], value->x, value->y, value->z);
//          printf("Set %s to (%f,%f,%f)\n", input.name,value->x, value->y, value->z);
          break;
        }
      case IN_INT:
        {
          const int* value = (int*) input.value_ptr;
          glUniform1i(inputs_[i], *value);
          break;
        }
      case IN_FLOAT:
        {
          const float* value = (float*) input.value_ptr;
          glUniform1f(inputs_[i], *value);
          break;
        }
    }
  }

  // launch compute shaders!
  glDispatchCompute(width_ / x_batch, height_, 1 );

  // prevent sampling befor all writes to image are done
  glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

//  glClear( GL_COLOR_BUFFER_BIT );
  glUseProgram( quad_program );
  frame_num += max_rays;
  glUniform1f(post_processor_mul_, 1.f/frame_num);
  glBindVertexArray( quad_vao );
  glActiveTexture( GL_TEXTURE0 );
  glBindTexture( GL_TEXTURE_2D, tex_output );
  glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
  SDL_GL_SwapWindow(window_);

  if ((frame_num & (frame_num - 1)) == 0 && frame_num > 8) {
    printf("Frame num: %d\n", frame_num);
    fflush(stdout);
  }
}

