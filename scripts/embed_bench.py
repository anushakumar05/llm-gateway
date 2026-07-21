import statistics
import time

from gateway.embeddings import embed_sync

embed_sync("warmup")

times = []
for i in range(100):
    t0 = time.perf_counter()
    embed_sync(f"how does database indexing work {i}")
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"n=100")
print(f"p50 {times[50]:.2f} ms")
print(f"p95 {times[95]:.2f} ms")
print(f"mean {statistics.mean(times):.2f} ms")