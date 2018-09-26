TARGETS=ray shader.comp ray.ptx exception.ptx anyhit.ptx

all: ${TARGETS}

clean: build/clean
	${RM} ${TARGETS}

.PHONY: FORCE
FORCE:

build/%: FORCE
	make -C build $*


${TARGETS}:%:build/%
	cp build/$@ $@
