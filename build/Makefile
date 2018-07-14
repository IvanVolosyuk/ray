PROF=-ggdb -pg -no-pie
PROF_GEN=-fprofile-generate
PROF_USE=-fprofile-use
LDFLAGS=-L/usr/local/X11R6/lib -Lc:\msys64\mingw64\lib -L/c/cygwin/home/Ivan/1/SDL2-2.0.8/x86_64-w64-mingw32/lib -lSDL2main -lSDL2 -lopengl32 -lglew32 -lpthread $(PROF_USE)
CFLAGS+=-I../src -ggdb
CXXFLAGS+=-I../src -ggdb -I/usr/include/SDL2 -std=c++17 -O3 -Wall $(PROF_USE) -Ic:\MinGW-w64\mingw64\include -Ic:\msys64\mingw64\include
OBJ=vector.o ray.o gl_renderer.o sw_renderer.o

vpath %.cpp ../src
vpath %.hpp ../src
vpath %.h ../src/shader
.PHONY: all
.PHONY: clean

all: ray

ray: $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

vector.o: vector.hpp

gl_renderer.o: gl_renderer.hpp vector.hpp renderer.hpp input.h

sw_renderer.o: sw_renderer.hpp renderer.hpp common.h shader.h software.h recursion.h stages.h input.h

ray.o: gl_renderer.hpp vector.hpp

%.pp : %.h
	gcc -DUSE_HW -E $< > $@

main.pp: main.h recursion.h stages.h shader.h software.h common.h input.h

shader.comp.in: main.pp
	sed  's/#/\/\/ GENERATED DONT EDIT/' $< > $@

shader.comp: shader.comp.in
	echo '#version 430' >$@
	cat $< >> $@

clean:
	$(RM) ray $(OBJ) shader.comp 

