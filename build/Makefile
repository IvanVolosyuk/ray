PROF=-ggdb -pg -no-pie
PROF_GEN=-fprofile-generate
PROF_USE=-fprofile-use
LDFLAGS=-L/usr/local/X11R6/lib -lGL -lglut -lSDL2 -lpthread $(PROF_USE)
CFLAGS+=-I../src -ggdb
CXXFLAGS+=-I../src -ggdb -I/usr/include/SDL2 -std=c++17 -O3 -Wall $(PROF_USE)
OBJ=vector.o ray.o gl_renderer.o

vpath %.cpp ../src
vpath %.hpp ../src
vpath %.h ../src/shader
.PHONY: all
.PHONY: clean

all: ray

ray: $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

vector.o: vector.hpp

gl_renderer.o: gl_renderer.hpp vector.hpp

ray.o: gl_renderer.hpp vector.hpp

%.pp : %.h
	gcc -E $< > $@

main.pp: main.h recursion.h stages.h

shader.comp: main.pp
	echo '#version 430' >$@
	sed  's/#/\/\/ GENERATED DONT EDIT/' $< >> $@

clean:
	$(RM) ray $(OBJ) shader.comp 

