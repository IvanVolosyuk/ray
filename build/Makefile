PROF=-ggdb -pg -no-pie
PROF_GEN=-fprofile-generate
PROF_USE=-fprofile-use
LDFLAGS=-L/usr/local/X11R6/lib -lGL -lglut -lSDL2 -lpthread $(PROF_USE)
CFLAGS+=-I../src -ggdb
CXXFLAGS+=-I../src -ggdb -I/usr/include/SDL2 -std=c++17 -O3 -Wall $(PROF_USE)
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

