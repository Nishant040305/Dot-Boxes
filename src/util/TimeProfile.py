
import time
from collections import defaultdict

class TimeProfile:
    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.starts = {}

    def start(self, name):
        self.starts[name] = time.perf_counter()

    def stop(self, name):
        if name in self.starts:
            duration = time.perf_counter() - self.starts[name]
            self.timings[name] += duration
            self.counts[name] += 1
            del self.starts[name]
    
    def interval(self, name):
        return _Interval(self, name)

    def report(self):
        """Returns a formatted string of timing statistics."""
        if not self.timings:
            return "No timing data."
        
        lines = []
        # Sort by total time descending
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        lines.append(f"{'Name':<25} | {'Total (s)':<10} | {'Count':<8} | {'Avg (ms)':<10}")
        lines.append("-" * 60)
        
        for name, total_time in sorted_items:
            count = self.counts[name]
            avg_ms = (total_time / count) * 1000 if count > 0 else 0
            lines.append(f"{name:<25} | {total_time:<10.4f} | {count:<8} | {avg_ms:<10.4f}")
            
        return "\n".join(lines)
        
    def reset(self):
        self.timings.clear()
        self.counts.clear()
        self.starts.clear()

class _Interval:
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        if self.profiler:
            self.profiler.start(self.name)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler:
            self.profiler.stop(self.name)
