.PHONY: check check-lock check-ruff check-ty check-test check-proto check-package test proto build

check: check-lock check-ruff check-ty check-test check-proto check-package

check-lock:
	uv lock --check

check-ruff:
	uv run --extra dev ruff check src tests scripts
	uv run --extra dev ruff format --check src tests scripts

check-ty:
	uv run --extra dev ty check

check-test:
	uv run --extra dev pytest --cov=src --cov-report=term-missing

check-proto: proto
	git diff --exit-code -- src/agentgateway_extproc/gen

check-package:
	@tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	uv build --out-dir "$$tmp"; \
	uv run python scripts/verify_distribution.py "$$tmp"

test: check-test

proto:
	rm -rf src/agentgateway_extproc/gen
	mkdir -p src/agentgateway_extproc/gen
	GOOGLE_DIR=$$(uv run python -c "import grpc_tools; print(grpc_tools.__path__[0] + '/_proto')"); \
	uv run python -m grpc_tools.protoc -Iprotos -I$$GOOGLE_DIR \
		--python_out=src/agentgateway_extproc/gen \
		--grpc_python_out=src/agentgateway_extproc/gen \
		--pyi_out=src/agentgateway_extproc/gen protos/*.proto
	@find src/agentgateway_extproc/gen -name '*.py' -exec sed -i.bak \
		-e 's/^import shared_envoy_pb2/from agentgateway_extproc.gen import shared_envoy_pb2/' \
		-e 's/^import ext_proc_pb2/from agentgateway_extproc.gen import ext_proc_pb2/' {} +
	@find src/agentgateway_extproc/gen -name '*.bak' -exec rm -f {} +
	@printf '%s\n' '"""Generated Envoy ext_proc protobuf bindings."""' \
		> src/agentgateway_extproc/gen/__init__.py

build:
	docker build -t agentgateway-extproc:local .
