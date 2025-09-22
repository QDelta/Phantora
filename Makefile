ifeq ($(CUDA_HOME),)
  $(error CUDA_HOME is not set)
endif

CC = gcc
CFLAGS = -isystem $(CUDA_HOME)/include -I include -I phantora/target/include -O2 -Wall -std=c11 -fPIC -fno-strict-aliasing -pthread 
TARGETS = build/phantora build/libcuda.so.1 build/visualizer

STATIC_LIBS = phantora/target/lib/libphantora.a

STUB_SRC = cudart cudart_noimpl cuda cublas cublas_noimpl nccl cudnn nvml

dist: all
	@mkdir -p dist
	@cp -a build/phantora dist/phantora_server
	@cp -a build/visualizer dist/phantora_visualizer
	@cp -a build/libcuda.so.1 dist/libcuda.so.1
	@ln -sf libcuda.so.1 dist/libcudart.so
	@ln -sf libcuda.so.1 dist/libcublas.so.11
	@ln -sf libcuda.so.1 dist/libcudnn.so.8
	@ln -sf libcuda.so.1 dist/libnvidia-ml.so.1

all:
	@$(MAKE) -C phantora
	@$(MAKE) $(TARGETS)

build/libcuda.so.1: $(STUB_SRC:%=build/%.o) $(STATIC_LIBS)
	@echo "LINK libcuda.so.1"
	@$(CC) $(CFLAGS) -shared -o $@ $^ -ldl -lm

build/phantora: phantora/target/bin/phantora
	@cp -a $< $@

build/visualizer: phantora/target/bin/visualizer
	@cp -a $< $@

build/%.o: stub/%.c
	@echo "CC   $<"
	@$(CC) $(CFLAGS) -c -o $@ $<

fmt:
	@$(MAKE) -C phantora fmt
	@clang-format -i --style="{BasedOnStyle: Mozilla, IndentWidth: 4}" stub/*.c include/common.h

clean:
	@$(MAKE) -C phantora clean
	@rm -f $(STUB_SRC:%=build/%.o) $(TARGETS)
	@rm -f dist/phantora_server dist/phantora_visualizer dist/libcuda.so.1 dist/libcudart.so dist/libcublas.so.11 dist/libnvidia-ml.so.1 dist/libcudnn.so.8
