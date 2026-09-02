.PHONY: install test benchmark docker clean

install:
	python -m pip install -e '.[test]'

test:
	pytest

benchmark:
	CPU_THREADS=$${CPU_THREADS:-1} OPENBLAS_NUM_THREADS=$${CPU_THREADS:-1} OMP_NUM_THREADS=$${CPU_THREADS:-1} python -m dflash_mini_lab.cli --output-dir reports

docker:
	docker build -t dflash-mini-lab .
	docker run --rm -e CPU_THREADS=$${CPU_THREADS:-1} -e OPENBLAS_NUM_THREADS=$${CPU_THREADS:-1} -e OMP_NUM_THREADS=$${CPU_THREADS:-1} -v "$$(pwd)/reports:/app/reports" dflash-mini-lab

clean:
	rm -f reports/benchmark.json reports/report.html reports/speedup.gif reports/architecture.gif
