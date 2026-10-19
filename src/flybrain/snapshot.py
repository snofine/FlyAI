from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flybrain.body.fly import FlyBody
from flybrain.brain.simulator import BrainSimulator
from flybrain.world.world import World


def save_snapshot(path: str | Path, brain: BrainSimulator, fly: FlyBody, world: World) -> None:
    target = Path(path)
    np.savez_compressed(
        target,
        voltage=brain.state.voltage,
        refractory=brain.state.refractory,
        spikes=brain.state.spikes,
        last_spike_step=brain.state.last_spike_step,
        pending_input=brain.pending_input,
        step_number=np.asarray(brain.step_number, dtype=np.int64),
        fly=np.asarray([fly.x, fly.y, fly.angle, fly.velocity, fly.angular_velocity]),
        threat=np.asarray([world.threat.x, world.threat.y, world.threat.radius, world.threat.speed]),
        metadata=np.asarray(json.dumps({"format": 1})),
    )


def load_snapshot(path: str | Path, brain: BrainSimulator, fly: FlyBody, world: World) -> None:
    with np.load(path) as data:
        if len(data["voltage"]) != brain.connectome.neuron_count:
            raise ValueError("Snapshot belongs to a different connectome")
        brain.state.voltage[:] = data["voltage"]
        brain.state.refractory[:] = data["refractory"]
        brain.state.spikes[:] = data["spikes"]
        brain.state.last_spike_step[:] = data["last_spike_step"]
        brain.pending_input[:] = data["pending_input"]
        brain.step_number = int(data["step_number"])
        fly.x, fly.y, fly.angle, fly.velocity, fly.angular_velocity = map(float, data["fly"])
        world.threat.x, world.threat.y, world.threat.radius, world.threat.speed = map(float, data["threat"])

