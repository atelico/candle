.PHONY: clean-ptx clean test

clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > ebcandle-kernels/src/lib.rs
	touch ebcandle-kernels/build.rs
	touch ebcandle-examples/build.rs
	touch ebcandle-flash-attn/build.rs

clean:
	cargo clean

test:
	cargo test

all: test
