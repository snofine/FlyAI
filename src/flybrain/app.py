from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from flybrain.body.fly import FlyBody
from flybrain.brain.connectome import Connectome
from flybrain.brain.neuron import LIFModel
from flybrain.brain.simulator import BrainSimulator
from flybrain.config import PROJECT_ROOT, load_config, load_yaml
from flybrain.experiment import ExperimentLog
from flybrain.motor.motor_decoder import MotorDecoder
from flybrain.sensory.stimulus_encoder import StimulusEncoder
from flybrain.sensory.manual import ManualStimulusController
from flybrain.sensory.vision import EyeSignals, VisionSensor
from flybrain.snapshot import load_snapshot, save_snapshot
from flybrain.world.world import World


def group_activity(active: np.ndarray, roles: dict[str, np.ndarray]) -> dict[str, float]:
    result = {}
    for name in ("visual", "central", "descending", "vnc", "motor"):
        group = roles.get(name, np.array([], dtype=np.int64))
        result[name] = float(np.count_nonzero(active[group]) / len(group)) if len(group) else 0.0
    return result


def build_runtime(config: dict, processed: Path, force_demo: bool):
    if not force_demo:
        try:
            connectome = Connectome.load(processed)
        except FileNotFoundError:
            connectome = Connectome.demo()
    else:
        connectome = Connectome.demo()
    neuron = config["neuron"]
    model = LIFModel(neuron["tau_ms"], neuron["threshold"], neuron["reset"], neuron["refractory_ms"])
    brain = BrainSimulator(connectome, model, config["simulation"]["dt_ms"])
    sensory_config = config.get("sensory") or load_yaml(PROJECT_ROOT / "config" / "sensory_neurons.yaml")
    looming_config = sensory_config.get("looming", {})
    looming_gain = float(looming_config.get("current_gain", 1.35))
    looming_baseline = float(looming_config.get("baseline_current", 0.0))
    encoder = StimulusEncoder(
        connectome.neuron_count,
        connectome.roles.get("looming", []),
        looming_gain,
        looming_baseline,
    )
    window_steps = max(1, int(config["motor"]["window_ms"] / brain.dt_ms))
    motor = MotorDecoder(connectome.roles, window_steps, config["motor"]["gain"])
    world_cfg = config["world"]
    world = World(world_cfg["width"], world_cfg["height"], world_cfg["threat_speed"])
    body_cfg = config["body"]
    fly = FlyBody(world.width * 0.35, world.height * 0.5, max_speed=body_cfg["max_speed"], turn_rate=body_cfg["turn_rate"], drag=body_cfg["drag"])
    return connectome, brain, encoder, motor, world, fly


def run(args: argparse.Namespace, config_override: dict | None = None) -> int:
    config = config_override or load_config(args.config)
    config.setdefault("sensory", load_yaml(PROJECT_ROOT / "config" / "sensory_neurons.yaml"))
    np.random.seed(int(config["simulation"]["seed"]))
    connectome, brain, encoder, decoder, world, fly = build_runtime(config, args.processed, args.demo)
    manual_config = config["sensory"].get("manual_stimuli", {})
    manual = ManualStimulusController(
        connectome.neuron_count, connectome.roles, manual_config, brain.dt_ms
    )
    if config_override is None:
        recorded_events = [
            {
                "time_ms": float(args.stimulus_at * 1000.0),
                "name": name,
                "duration_ms": float(args.stimulus_duration),
                "strength": 1.0,
            }
            for name in args.stimulate
        ]
        config["experiment"] = {"manual_events": recorded_events}
    else:
        recorded_events = list(config.get("experiment", {}).get("manual_events", []))
        config.setdefault("experiment", {})["manual_events"] = recorded_events
    scheduled_events = [dict(event) for event in recorded_events]
    fired_scheduled_events: set[int] = set()
    if args.snapshot and args.snapshot.exists():
        load_snapshot(args.snapshot, brain, fly, world)

    if connectome.is_demo:
        print("NOTICE: processed MaleCNS data not found; running the explicit 8-neuron demo circuit.")
        print("Run 'uv run flybrain-download' then 'uv run flybrain-preprocess' for the real graph.")
    else:
        print(f"Loaded MaleCNS: {connectome.neuron_count:,} neurons, {connectome.edge_count:,} edges")

    experiment_name = args.experiment or datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    build_info_path = args.processed / "build_info.json"
    build_info = json.loads(build_info_path.read_text(encoding="utf-8")) if build_info_path.exists() and not connectome.is_demo else {"demo": True}
    logger = ExperimentLog(
        args.log_dir / experiment_name,
        {
            "seed": config["simulation"]["seed"],
            "config": config,
            "demo": connectome.is_demo,
            "connectome": build_info,
        },
    )
    vision = VisionSensor()
    renderer = None
    if not args.headless:
        from flybrain.ui.renderer import Renderer
        renderer = Renderer(world.width, world.height, args.debug)

    duration_steps = int(args.seconds * 1000 / brain.dt_ms) if args.seconds > 0 else None
    physics_interval = max(1, int(round(1000 / config["world"]["physics_hz"] / brain.dt_ms)))
    log_interval = max(1, int(round(config["logging"]["sample_every_ms"] / brain.dt_ms)))
    signals = EyeSignals()
    command = decoder.update(brain.state.spikes)
    latest = None
    running = True
    started = time.perf_counter()
    try:
        while running and (duration_steps is None or brain.step_number < duration_steps):
            if renderer:
                running, reset, triggered = renderer.events(fly)
                if reset:
                    world.reset_threat(fly)
                for name in triggered:
                    manual.trigger(name, args.stimulus_duration)
                    recorded_events.append({
                        "time_ms": brain.time_ms,
                        "name": name,
                        "duration_ms": float(args.stimulus_duration),
                        "strength": 1.0,
                    })
            for _ in range(config["simulation"]["neural_steps_per_frame"] if renderer else 1):
                dt_s = brain.dt_ms / 1000.0
                for event_index, event in enumerate(scheduled_events):
                    if event_index not in fired_scheduled_events and brain.time_ms >= float(event["time_ms"]):
                        manual.trigger(
                            str(event["name"]),
                            float(event.get("duration_ms", args.stimulus_duration)),
                            float(event.get("strength", 1.0)),
                        )
                        fired_scheduled_events.add(event_index)
                signals = vision.sense(fly.x, fly.y, fly.angle, world.threat.x, world.threat.y, world.threat.radius, dt_s)
                contact_strength = world.contact_strength(fly)
                if contact_strength > 0 and "touch_head" in manual.available_channels:
                    manual.trigger("touch_head", duration_ms=30.0, strength=contact_strength)
                sensory_current = encoder.encode_looming(signals.looming) + manual.encode()
                latest = brain.step(sensory_current)
                command = decoder.update(latest.spikes)
                world.update(fly, dt_s)
                if brain.step_number % physics_interval == 0:
                    physics_dt = physics_interval * dt_s
                    fly.update(command, physics_dt, world.width, world.height)
                if brain.step_number % log_interval == 0:
                    logger.append({
                        "time_ms": brain.time_ms, "looming": signals.looming,
                        "active_neurons": latest.active_count, "spike_count": latest.spike_count,
                        "forward": command.forward, "turn_left": command.turn_left,
                        "turn_right": command.turn_right, "escape": command.escape,
                        "fly_x": fly.x, "fly_y": fly.y, "fly_angle": fly.angle,
                        "threat_x": world.threat.x, "threat_y": world.threat.y,
                        "contact_strength": contact_strength,
                        "manual_stimulus_count": len(manual.active_names()),
                    })
                if duration_steps is not None and brain.step_number >= duration_steps:
                    break
            if renderer and latest:
                recent = brain.state.last_spike_step >= brain.step_number - max(1, int(100 / brain.dt_ms))
                recent_indices = np.flatnonzero(recent)
                if len(recent_indices):
                    recent_order = np.argsort(brain.state.last_spike_step[recent_indices])[::-1]
                    recent_indices = recent_indices[recent_order[:8]]
                telemetry = {
                    "is_demo": connectome.is_demo, "time_ms": brain.time_ms,
                    "neurons": connectome.neuron_count, "edges": connectome.edge_count,
                    "spikes": latest.spike_count, "active": latest.active_count,
                    "vision_l": signals.brightness_left, "vision_r": signals.brightness_right,
                    "looming": signals.looming, "forward": command.forward,
                    "turn_left": command.turn_left, "turn_right": command.turn_right,
                    "escape": command.escape, "brain_groups": group_activity(recent, connectome.roles),
                    "active_body_ids": connectome.neuron_ids[recent_indices].tolist(),
                    "manual_stimuli": ",".join(manual.active_names()) or "-",
                }
                renderer.draw(fly, world, telemetry)
                renderer.tick(60)
    finally:
        if renderer:
            renderer.close()
        output = logger.save()
        if args.snapshot:
            save_snapshot(args.snapshot, brain, fly, world)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "simulated_seconds": brain.time_ms / 1000.0,
        "wall_seconds": round(elapsed, 3),
        "neural_steps_per_second": round(brain.step_number / max(elapsed, 1e-9), 1),
        "final_position": [round(fly.x, 3), round(fly.y, 3)],
        "log": str(output),
    }, indent=2))
    return 0


def replay(args: argparse.Namespace) -> int:
    manifest, expected = ExperimentLog.load(args.replay)
    if expected.empty:
        raise ValueError("Cannot replay an empty experiment log")
    replay_name = f"{args.replay.name}_replay"
    replay_args = argparse.Namespace(**vars(args))
    replay_args.replay = None
    replay_args.demo = bool(manifest.get("demo", False))
    replay_args.seconds = float(expected["time_ms"].max()) / 1000.0
    replay_args.experiment = replay_name
    replay_args.snapshot = None
    exit_code = run(replay_args, manifest["config"])
    _, actual = ExperimentLog.load(args.log_dir / replay_name)
    comparable = list(expected.columns)
    verified = len(expected) == len(actual) and np.allclose(
        expected[comparable].to_numpy(dtype=float),
        actual[comparable].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )
    print(json.dumps({"replay_verified": bool(verified), "rows": len(actual)}, indent=2))
    return exit_code if verified else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Virtual Drosophila driven through a MaleCNS-shaped neural graph")
    result.add_argument("--headless", action="store_true")
    result.add_argument("--debug", action="store_true")
    result.add_argument("--demo", action="store_true", help="Force the bundled 8-neuron demonstrator")
    result.add_argument("--seconds", type=float, default=0.0, help="Stop after virtual seconds (required for headless)")
    result.add_argument("--config", type=Path)
    result.add_argument("--processed", type=Path, default=PROJECT_ROOT / "data" / "processed")
    result.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")
    result.add_argument("--experiment")
    result.add_argument("--replay", type=Path)
    result.add_argument("--snapshot", type=Path)
    result.add_argument(
        "--stimulate", action="append", default=[],
        choices=("eye_left", "eye_right", "leg_front", "leg_middle", "leg_hind", "touch_head"),
        help="Manually stimulate a real sensory-neuron group; may be repeated",
    )
    result.add_argument("--stimulus-at", type=float, default=0.0, help="Manual stimulus time in virtual seconds")
    result.add_argument("--stimulus-duration", type=float, default=180.0, help="Manual stimulus duration in ms")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.replay:
        return replay(args)
    if args.headless and args.seconds <= 0:
        args.seconds = 5.0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
