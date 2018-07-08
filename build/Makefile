LDFLAGS=-L/usr/local/X11R6/lib -lGL -lglut -lSDL2 -lpthread
CFLAGS+=-I../src -ggdb
CXXFLAGS+=-I../src -ggdb -I/usr/include/SDL2 -std=c++17 -O3

vpath %.cpp ../src
vpath %.hpp ../src
.PHONY: all
.PHONY: clean

all: ray

ray: vector.o ray.o
	$(CXX) -o ray vector.o ray.o $(LDFLAGS)

vector.o: vector.hpp

clean:
	$(RM) ray ray.o vector.o

