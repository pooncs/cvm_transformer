import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt


class TelemetryDashboard:
    def __init__(self):
        self.data = defaultdict(list)

    def log(self, key, value):
        self.data[key].append((time.time(), value))

    def plot(self, keys, save_path="telemetry.png"):
        for k in keys:
            xs, ys = zip(*self.data[k])
            plt.plot(xs, ys, label=k)
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def export_json(self, path="telemetry.json"):
        with open(path, "w") as f:
            json.dump({k: [(t, v) for t, v in vals] for k, vals in self.data.items()}, f)


if __name__ == "__main__":
    dash = TelemetryDashboard()
    for i in range(10):
        dash.log("latency_ms", 100 + i * 5)
        time.sleep(0.1)
    dash.plot(["latency_ms"])