PROF=-ggdb -pg -no-pie
PROF_GEN=-fprofile-generate
PROF_USE=-fprofile-use
OPT=-momit-leaf-frame-pointer \
	  -mtune=native -pipe -march=native -mmmx -msse -msse2 \
	  -msse3 -mssse3 -mfpmath=sse
# Win64
ifeq ($(OS),Windows_NT)
LDFLAGS=-L. -L/usr/lib/w32api -lmingw32 -lSDL2main -lSDL2 -L/c/cygwin/home/Ivan/1/SDL2-2.0.8/x86_64-w64-mingw32/lib -lglew32 -lopengl32 -lpng
else
LDFLAGS=-L. -lSDL2 -lGL -lSDL2 -lGLEW -lpthread -lpng -L/home/ivan/experiments/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64 -loptix -loptixu
endif
CFLAGS+=-I../src -ggdb
CXXFLAGS+=-I../src -I/usr/include/SDL2 -std=c++17 \
	  -O3 $(OPT) -Wall \
	  -Ic:\MinGW-w64\mingw64\include \
	  -Ic:\msys64\mingw64\include \
	  -DIMGUI_IMPL_OPENGL_LOADER_GLEW \
	  -I../src/imgui -I/home/ivan/experiments/NVIDIA-OptiX-SDK-5.1.0-linux64/include \
	  -I/usr/local/cuda/include
OBJ=vector.o ray.o sw_renderer.o gl_renderer.o png.o texture.o imgui_demo.o imgui_draw.o imgui_widgets.o imgui.o imgui_impl_opengl3.o imgui_impl_sdl.o

vpath %.cpp ../src ../src/imgui  ../src/imgui/examples
vpath %.hpp ../src ../src/shader
vpath %.cu ../src/optix
.PHONY: all
.PHONY: clean

all: ray shader.comp ray.ptx triangle.ptx exception.ptx anyhit.ptx

ray: $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

vector.o: vector.hpp

gl_renderer.o: gl_renderer.hpp vector.hpp renderer.hpp input.hpp

sw_renderer.o: sw_renderer.hpp renderer.hpp common.hpp shader.hpp software.hpp input.hpp texture.hpp kd.hpp

png.o: png.hpp

ray.o: gl_renderer.hpp vector.hpp texture.hpp

texture.o: texture.hpp

%.pp : %.h
	gcc -DUSE_HW -E $< > $@

main.pp: main.hpp shader.hpp software.hpp common.hpp input.hpp png.hpp

shader.comp.in: main.pp
	sed  's/#/\/\/ GENERATED DONT EDIT/' $< > $@

shader.comp: shader.comp.in
	echo '#version 430' >$@
	cat $< >> $@

%.ptx : %.cu
	/usr/local/cuda/bin/nvcc --ptx --maxrregcount 1000 --compile --generate-line-info $< --include-path ../src,/home/ivan/experiments/NVIDIA-OptiX-SDK-5.1.0-linux64/include

clean:
	$(RM) ray $(OBJ) shader.comp 


