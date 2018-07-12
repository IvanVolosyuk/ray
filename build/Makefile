PROF=-ggdb -pg -no-pie
PROF_GEN=-fprofile-generate
PROF_USE=-fprofile-use
LDFLAGS=-L/usr/local/X11R6/lib -lGL -lglut -lSDL2 -lpthread $(PROF_USE)
CFLAGS+=-I../src -ggdb
CXXFLAGS+=-I../src -ggdb -I/usr/include/SDL2 -std=c++17 -O3 $(PROF_USE)

vpath %.cpp ../src
vpath %.hpp ../src
vpath %.h ../src/shader
.PHONY: all
.PHONY: clean

all: ray

ray: vector.o ray.o gl_renderer.o
	$(CXX) -o ray vector.o ray.o gl_renderer.o $(LDFLAGS)

vector.o: vector.hpp

gl_renderer.o: gl_renderer.hpp

%.pp : %.h
	gcc -E $< > $@

main.pp: main.h recursion.h stages.h

shader.comp: main.pp
	echo '#version 430' >$@
	sed  's/#/\/\/ GENERATED DONT EDIT/' $< >> $@

clean:
	$(RM) ray ray.o vector.o

