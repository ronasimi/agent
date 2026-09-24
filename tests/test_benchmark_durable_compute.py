from diagnostics.benchmarks.benchmark_durable_compute import run_benchmark


def test_durable_compute_benchmark_reports_throughput_and_checkpoint_cost():
    report = run_benchmark(quanta=(100,), runs=1, checkpoint_runs=2)
    row = report["quantum_results"][0]
    assert row["quantum"] == 100
    assert row["median_transitions_per_second"] > 0
    assert report["checkpoint_write"]["runs"] == 2
    assert report["checkpoint_write"]["median_ms"] >= 0
