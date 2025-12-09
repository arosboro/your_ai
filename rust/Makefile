.PHONY: build test run-example check fmt clippy clean help

help:
	@echo "Empirical Distrust Training - Rust Implementation"
	@echo ""
	@echo "Available commands:"
	@echo "  make build        - Build release binary"
	@echo "  make test         - Run all tests"
	@echo "  make run-example  - Run basic training example"
	@echo "  make check        - Check code without building"
	@echo "  make fmt          - Format code"
	@echo "  make clippy       - Run linter"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make setup        - Run hardware setup"
	@echo "  make recommend    - Show model recommendations"

build:
	cargo build --release

test:
	cargo test

run-example:
	cargo run --example basic_training

check:
	cargo check

fmt:
	cargo fmt

clippy:
	cargo clippy -- -D warnings

clean:
	cargo clean

setup:
	cargo run --bin your_ai -- setup

recommend:
	cargo run --bin your_ai -- recommend

# Development helpers
watch:
	cargo watch -x check -x test

doc:
	cargo doc --open

bench:
	cargo bench

