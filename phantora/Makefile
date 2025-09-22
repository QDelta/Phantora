all:
	@cargo build --release
	@mkdir -p target/include target/lib target/bin
	@cp target/release/simulator target/bin/phantora
	@cp target/release/visualizer target/bin/visualizer
	@cp target/release/libcuda_call.a target/lib/libphantora.a
	@echo "Generating C header..."
	@cbindgen --quiet --crate cuda_call --output target/include/phantora.h

fmt:
	@cargo fmt

clean:
	@cargo clean
