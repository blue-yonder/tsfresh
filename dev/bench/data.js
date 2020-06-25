window.BENCHMARK_DATA = {
  "lastUpdate": 1593078875567,
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
      },
      {
        "commit": {
          "author": {
            "name": "blue-yonder",
            "username": "blue-yonder"
          },
          "committer": {
            "name": "blue-yonder",
            "username": "blue-yonder"
          },
          "id": "97bea63a09346e3c098912f6be19761fb63b1553",
          "message": "Add benchmarking test",
          "timestamp": "2020-06-08T09:07:28Z",
          "url": "https://github.com/blue-yonder/tsfresh/pull/710/commits/97bea63a09346e3c098912f6be19761fb63b1553"
        },
        "date": 1591648921084,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark.py::test_benchmark_small_data",
            "value": 0.2001063187483521,
            "unit": "iter/sec",
            "range": "stddev: 0.07774186298673909",
            "extra": "mean: 4.997343443500007 sec\nrounds: 2"
          },
          {
            "name": "tests/benchmark.py::test_benchmark_large_data",
            "value": 0.34781510271361843,
            "unit": "iter/sec",
            "range": "stddev: 0.016463188884254368",
            "extra": "mean: 2.875090794500011 sec\nrounds: 2"
          },
          {
            "name": "tests/benchmark.py::test_benchmark_with_selection",
            "value": 0.2097995181180121,
            "unit": "iter/sec",
            "range": "stddev: 0.0009091646652443717",
            "extra": "mean: 4.766455180499989 sec\nrounds: 2"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "blue-yonder",
            "username": "blue-yonder"
          },
          "committer": {
            "name": "blue-yonder",
            "username": "blue-yonder"
          },
          "id": "c40a057d41458303135fdde13c0cd0b8933400c9",
          "message": "Add benchmarking test",
          "timestamp": "2020-06-25T08:06:27Z",
          "url": "https://github.com/blue-yonder/tsfresh/pull/710/commits/c40a057d41458303135fdde13c0cd0b8933400c9"
        },
        "date": 1593078875111,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark.py::test_benchmark_small_data",
            "value": 0.20656169377173203,
            "unit": "iter/sec",
            "range": "stddev: 0.009986355226502463",
            "extra": "mean: 4.841168668499996 sec\nrounds: 2"
          },
          {
            "name": "tests/benchmark.py::test_benchmark_large_data",
            "value": 0.3767772633671491,
            "unit": "iter/sec",
            "range": "stddev: 0.029941877327895133",
            "extra": "mean: 2.6540879645000075 sec\nrounds: 2"
          },
          {
            "name": "tests/benchmark.py::test_benchmark_with_selection",
            "value": 0.2265800912702651,
            "unit": "iter/sec",
            "range": "stddev: 0.0011348349660196216",
            "extra": "mean: 4.413450424499999 sec\nrounds: 2"
          }
        ]
      }
    ]
  }
}