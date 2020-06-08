window.BENCHMARK_DATA = {
  "lastUpdate": 1591648586475,
  "repoUrl": "https://github.com/blue-yonder/tsfresh",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "nilslennartbraun@gmail.com",
            "name": "Nils Braun",
            "username": "nils-braun"
          },
          "committer": {
            "email": "nilslennartbraun@gmail.com",
            "name": "Nils Braun",
            "username": "nils-braun"
          },
          "distinct": true,
          "id": "98608bfe3491374ac18363f65a0fedeca45341fe",
          "message": "Speed up the tests",
          "timestamp": "2020-06-08T22:33:36+02:00",
          "tree_id": "c3c4b642e3927b34a1a7f9ab45f189123954534c",
          "url": "https://github.com/blue-yonder/tsfresh/commit/98608bfe3491374ac18363f65a0fedeca45341fe"
        },
        "date": 1591648585909,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark.py::test_benchmark_small_data",
            "value": 0.1886700983964568,
            "unit": "iter/sec",
            "range": "stddev: 0.195254537466511",
            "extra": "mean: 5.300256948500007 sec\nrounds: 2"
          },
          {
            "name": "tests/benchmark.py::test_benchmark_large_data",
            "value": 0.3255812162488828,
            "unit": "iter/sec",
            "range": "stddev: 0.0649351020548233",
            "extra": "mean: 3.071430260999989 sec\nrounds: 2"
          },
          {
            "name": "tests/benchmark.py::test_benchmark_with_selection",
            "value": 0.18915024613390544,
            "unit": "iter/sec",
            "range": "stddev: 0.13031391785748234",
            "extra": "mean: 5.286802530999978 sec\nrounds: 2"
          }
        ]
      }
    ]
  }
}