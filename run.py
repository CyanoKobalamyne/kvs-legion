#!/usr/bin/env python3
import os

count = 30

mem = 200
batch = 20

transactions = [1, 2, 5, 10, 20, 50, 100]
cpus = [1, 2, 4, 8, 12]

print("Memory size:", mem)
print("Batch size:", batch)
print()

print(" #rd  #wr  " + " ".join(f"{c:4d}" for c in cpus))
for r in transactions:
    for w in [0, r]:
        print(f"{r:4d} {w:4d}  ", end="")
        results = []
        for cpu in cpus:
            times = []
            for _ in range(count):
                stream = os.popen(
                    f"./db_test -m {mem} -b {batch} -r {r} -w {w} -ll:cpu {cpu} -level 5")
                times.append(int(stream.read().strip()))
            results.append(sum(times) / len(times))
        print(" ".join(f"{r/1e6:4.0f}" for r in results))
print()

print("#trf  " + " ".join(f"{c:4d}" for c in cpus))
for t in transactions:
    print(f"{t:4d}  ", end="")
    results = []
    for cpu in cpus:
        times = []
        for _ in range(count):
            stream = os.popen(
                f"./db_test -m {mem} -b {batch} -t {t} -ll:cpu {cpu} -level 5")
            times.append(int(stream.read().strip()))
        results.append(sum(times) / len(times))
    print(" ".join(f"{r/1e6:4.0f}" for r in results))
