#!/usr/bin/env python3
"""Run reproducible mono/stereo VINS benchmarks on one EuRoC sequence."""

import argparse
import csv
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


ORDER = ("mono", "stereo", "stereo", "mono", "mono", "stereo")
REQUIRED_COMMANDS = ("roscore", "rosrun", "rospack")


def command_path(name):
    return shutil.which(name)


def validate_environment(repo, dataset):
    missing = [name for name in REQUIRED_COMMANDS if not command_path(name)]
    if missing:
        raise RuntimeError("missing ROS1 commands: " + ", ".join(missing) +
                           "; source ROS and the catkin workspace first")
    required = [dataset / "mav0/cam0/data.csv", dataset / "mav0/cam1/data.csv",
                dataset / "mav0/imu0/data.csv",
                dataset / "mav0/state_groundtruth_estimate0/data.csv"]
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        raise RuntimeError("missing dataset files: " + ", ".join(missing_files))
    for package, executable in (("feature_tracker", "feature_tracker"),
                                ("vins_estimator", "vins_estimator")):
        result = subprocess.run(["rospack", "find", package], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError("ROS package is not discoverable: " + package)


def csv_data_count(path):
    with path.open() as stream:
        return sum(1 for line in stream if line.strip() and not line.startswith("#"))


def snapshot_metadata(repo, dataset, output):
    def run(*command):
        try:
            return subprocess.run(command, cwd=str(repo), text=True,
                                  capture_output=True).stdout.strip()
        except FileNotFoundError:
            return "unavailable (command not installed: %s)" % command[0]
    metadata = {
        "created_unix_s": time.time(),
        "dataset": str(dataset.resolve()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_status": run("git", "status", "--short"),
        "cam0_frames": csv_data_count(dataset / "mav0/cam0/data.csv"),
        "cam1_frames": csv_data_count(dataset / "mav0/cam1/data.csv"),
        "imu_samples": csv_data_count(dataset / "mav0/imu0/data.csv"),
    }
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    try:
        (output / "cpuinfo.txt").write_text(Path("/proc/cpuinfo").read_text())
    except OSError:
        pass


def materialize_config(source, destination, run_dir):
    text = source.read_text()
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.lstrip().startswith("output_path:"):
            lines.append('output_path: "' + str(run_dir.resolve()) + '/"')
            replaced = True
        elif line.lstrip().startswith(("cam0_calib:", "cam1_calib:")):
            key, value = line.split(":", 1)
            relative = value.strip().strip('"').strip("'")
            calibration = Path(relative)
            if not calibration.is_absolute():
                calibration = source.parent / calibration
            lines.append(f'{key}: "{calibration.resolve()}"')
        else:
            lines.append(line)
    if not replaced:
        raise RuntimeError("configuration has no output_path: " + str(source))
    destination.write_text("\n".join(lines) + "\n")


def descendants(pid):
    result, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        children_path = Path("/proc") / str(parent) / "task" / str(parent) / "children"
        try:
            children = [int(value) for value in children_path.read_text().split()]
        except (OSError, ValueError):
            children = []
        for child in children:
            if child not in result:
                result.add(child)
                pending.append(child)
    return {pid} | result


def process_stats(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split()
        rss_pages = int(fields[23])
        ticks = int(fields[13]) + int(fields[14])
        return ticks, rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class ResourceSampler(threading.Thread):
    def __init__(self, path, groups, interval=0.1):
        super().__init__(daemon=True)
        self.path = path
        self.groups = groups
        self.interval = interval
        self.stop_event = threading.Event()
        self.previous = {}

    def run(self):
        clock_ticks = os.sysconf("SC_CLK_TCK")
        previous_time = time.monotonic()
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("elapsed_s", "group", "process_count", "cpu_percent", "rss_bytes"))
            start = previous_time
            while not self.stop_event.wait(self.interval):
                now = time.monotonic()
                elapsed = max(now - previous_time, 1e-9)
                for name, process in self.groups.items():
                    pids = descendants(process.pid) if process.poll() is None else set()
                    total_ticks, total_rss = 0, 0
                    for pid in pids:
                        stats = process_stats(pid)
                        if stats:
                            total_ticks += stats[0]
                            total_rss += stats[1]
                    previous_ticks = self.previous.get(name, total_ticks)
                    cpu = 100.0 * (total_ticks - previous_ticks) / clock_ticks / elapsed
                    writer.writerow((f"{now-start:.6f}", name, len(pids), f"{max(cpu, 0):.6f}", total_rss))
                    self.previous[name] = total_ticks
                stream.flush()
                previous_time = now

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2.0)


def start_process(command, log_path, env):
    log = log_path.open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True, text=True)
    process._benchmark_log = log
    return process


def stop_process(process, timeout=10.0):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    process._benchmark_log.close()


def wait_for_stable_file(path, timeout=15.0, stable_for=2.0):
    deadline, stable_since, previous = time.monotonic() + timeout, None, None
    while time.monotonic() < deadline:
        current = path.stat().st_size if path.exists() else -1
        if current > 0 and current == previous:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= stable_for:
                return True
        else:
            stable_since = None
        previous = current
        time.sleep(0.25)
    return False


def run_once(repo, dataset, mode, run_dir, port):
    run_dir.mkdir(parents=True)
    config_source = repo / "config/euroc" / ("euroc_stereo_eval.yaml" if mode == "stereo" else "euroc_mono_eval.yaml")
    config = run_dir / "config.yaml"
    materialize_config(config_source, config, run_dir)
    env = os.environ.copy()
    env["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % port
    env["ROS_HOME"] = str((run_dir / "ros_home").resolve())
    env["VINS_BENCHMARK_DIR"] = str(run_dir.resolve())
    Path(env["ROS_HOME"]).mkdir()
    processes = {}
    sampler = None
    status = {"mode": mode, "success": False, "reason": "unknown", "port": port}
    try:
        processes["roscore"] = start_process(["roscore", "-p", str(port)], run_dir / "roscore.log", env)
        time.sleep(2.0)
        if processes["roscore"].poll() is not None:
            raise RuntimeError("roscore exited during startup")
        common = ["_config_file:=" + str(config.resolve()), "_vins_folder:=" + str(repo.resolve()) + "/"]
        processes["feature_tracker"] = start_process(
            ["rosrun", "feature_tracker", "feature_tracker", *common], run_dir / "feature_tracker.log", env)
        processes["vins_estimator"] = start_process(
            ["rosrun", "vins_estimator", "vins_estimator", *common], run_dir / "vins_estimator.log", env)
        time.sleep(3.0)
        if any(processes[name].poll() is not None for name in ("feature_tracker", "vins_estimator")):
            raise RuntimeError("a VINS node exited during startup")
        player_command = [sys.executable, str(repo / "tools/play_euroc_ros1.py"), str(dataset),
                          "--rate", "1.0", "--left-topic", "/cam0/image_raw",
                          "--imu-topic", "/imu0"]
        if mode == "stereo":
            player_command += ["--stereo", "--rectify-euroc", "--right-topic", "/cam1/image_raw"]
        processes["player"] = start_process(player_command, run_dir / "player.log", env)
        sampler = ResourceSampler(run_dir / "resources.csv", processes)
        sampler.start()
        player_code = processes["player"].wait()
        if player_code:
            raise RuntimeError("dataset player exited with code %d" % player_code)
        # Let the asynchronous stereo worker and estimator consume their final queues.
        time.sleep(3.0)
        if not wait_for_stable_file(run_dir / "vins_result_no_loop.csv"):
            raise RuntimeError("trajectory was not produced or backend did not drain")
        if any(processes[name].poll() is not None for name in ("feature_tracker", "vins_estimator")):
            raise RuntimeError("a VINS node exited before shutdown")
        estimator_log = (run_dir / "vins_estimator.log").read_text(errors="ignore")
        if "system reboot!" in estimator_log:
            raise RuntimeError("estimator rebooted")
        status.update(success=True, reason="complete")
    except Exception as error:
        status["reason"] = str(error)
    finally:
        if sampler:
            sampler.stop()
        for name in ("player", "feature_tracker", "vins_estimator", "roscore"):
            stop_process(processes.get(name))
        status["return_codes"] = {name: process.returncode for name, process in processes.items()}
        (run_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        with (run_dir / "run.log").open("w") as combined:
            for name in ("roscore", "feature_tracker", "vins_estimator", "player"):
                path = run_dir / (name + ".log")
                if path.exists():
                    combined.write("===== " + name + " =====\n")
                    combined.write(path.read_text(errors="ignore"))
                    combined.write("\n")
    return status["success"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-attempts-per-mode", type=int, default=6)
    parser.add_argument("--ros-port", type=int, default=11341)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    dataset = (args.dataset or repo / "datasets/MH_03_medium").resolve()
    output = (args.output or repo / "outputs/euroc_eval/MH_03_medium").resolve()
    try:
        validate_environment(repo, dataset)
    except RuntimeError as error:
        raise SystemExit("benchmark prerequisite check failed: " + str(error))
    if args.validate_only:
        print("benchmark prerequisites are available")
        return
    if args.runs <= 0 or args.max_attempts_per_mode < args.runs:
        raise SystemExit("--runs must be positive and max attempts must be at least runs")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_metadata(repo, dataset, output)
    desired_order = list(ORDER if args.runs == 3 else
                         [mode for _ in range(args.runs) for mode in ("mono", "stereo")])
    successes = {"mono": 0, "stereo": 0}
    attempts = {"mono": 0, "stereo": 0}
    queue = desired_order[:]
    while queue:
        mode = queue.pop(0)
        if successes[mode] >= args.runs:
            continue
        attempts[mode] += 1
        if attempts[mode] > args.max_attempts_per_mode:
            raise SystemExit("too many failed %s attempts" % mode)
        temporary = output / mode / ("attempt_%02d" % attempts[mode])
        print("[%s] attempt %d" % (mode, attempts[mode]), flush=True)
        if run_once(repo, dataset, mode, temporary, args.ros_port + attempts[mode] + (0 if mode == "mono" else 20)):
            successes[mode] += 1
            final = output / mode / ("run_%d" % successes[mode])
            temporary.rename(final)
        else:
            failed = output / mode / "failed_attempts"
            failed.mkdir(exist_ok=True)
            temporary.rename(failed / temporary.name)
            queue.append(mode)
    evaluator = repo / "tools/evaluate_euroc_benchmark.py"
    subprocess.run([sys.executable, str(evaluator), "--repo", str(repo),
                    "--dataset", str(dataset), "--output", str(output)], check=True)


if __name__ == "__main__":
    main()
