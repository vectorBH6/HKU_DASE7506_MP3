"""Apply reproduction limits before importing the official entry point."""
import argparse
import os
import resource
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--module", required=True)
parser.add_argument("--memory-gib", type=int, default=0)
args, rest = parser.parse_known_args()
if hasattr(os, "sched_getaffinity"):
    cpus = [int(c) for c in os.environ["MP3_CPU_AFFINITY"].split(",")] if "MP3_CPU_AFFINITY" in os.environ else sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise ValueError("Four CPU affinity entries are required")
    os.sched_setaffinity(0, cpus)
if args.memory_gib:
    limit = args.memory_gib * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
sys.argv = [args.module] + rest
runpy.run_module(args.module, run_name="__main__")
