from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flybrain.app import build_runtime
from flybrain.body.fly import FlyBody
from flybrain.brain.connectome import Connectome
from flybrain.brain.neuron import LIFModel, NeuronState
from flybrain.brain.neurotransmitters import transmitter_signs
from flybrain.brain.simulator import BrainSimulator
from flybrain.brain.synapses import normalize_weights
from flybrain.config import load_config
from flybrain.experiment import ExperimentLog
from flybrain.motor.motor_decoder import MotorDecoder
from flybrain.sensory.stimulus_encoder import StimulusEncoder
from flybrain.sensory.manual import ManualStimulusController
from flybrain.sensory.vision import VisionSensor
from flybrain.snapshot import load_snapshot, save_snapshot
from flybrain.tools.preprocess_connectome import preprocess
from flybrain.tools.inspect_neuron import metadata_value
from flybrain.world.world import World


def test_weight_normalization() -> None:
    values = normalize_weights(np.array([0, 1, 9]), "log1p", 0.5)
    assert values.dtype == np.float32
    assert np.allclose(values, np.log1p([0, 1, 9]) * 0.5)


def test_neurotransmitter_modes() -> None:
    metadata = pd.DataFrame({"body": [1, 2, 3], "consensus_nt": ["acetylcholine", "gaba", "dopamine"]})
    config_path = Path(__file__).parents[1] / "config" / "neurotransmitters.yaml"
    neutral = transmitter_signs(np.array([1, 2, 3]), metadata, "body", config_path, "neutral")
    experimental = transmitter_signs(np.array([1, 2, 3]), metadata, "body", config_path, "experimental")
    assert neutral.tolist() == [1.0, 1.0, 1.0]
    assert experimental.tolist() == [1.0, -1.0, 0.0]


def test_lif_spike_and_refractory() -> None:
    model = LIFModel(tau_ms=20, threshold=1, reset=0, refractory_ms=2)
    state = NeuronState.create(1)
    assert model.step(state, np.array([1.1], dtype=np.float32), 1, 0)[0]
    assert state.refractory[0] == 2
    assert not model.step(state, np.array([10], dtype=np.float32), 1, 1)[0]
    assert not model.step(state, np.array([10], dtype=np.float32), 1, 2)[0]
    assert model.step(state, np.array([1.1], dtype=np.float32), 1, 3)[0]


def test_event_driven_spike_propagation() -> None:
    graph = Connectome.demo(scale=0.5)
    brain = BrainSimulator(graph, LIFModel(refractory_ms=1), dt_ms=1)
    brain.step(np.array([1.1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert brain.state.spikes[0]
    assert brain.pending_input[2] > 0
    brain.step()
    assert brain.state.spikes[2]


def test_sensory_encoder_only_targets_sensory_group() -> None:
    graph = Connectome.demo()
    encoder = StimulusEncoder(graph.neuron_count, graph.roles["looming"], gain=2.0)
    current = encoder.encode_looming(0.75)
    assert np.count_nonzero(current) == 2
    assert np.all(current[graph.roles["looming"]] == 1.5)
    assert not np.any(current[graph.roles["motor"]])


def test_manual_stimulus_targets_only_selected_sensory_group() -> None:
    roles = {"stimulus_leg_front": np.array([1, 3], dtype=np.int64)}
    controller = ManualStimulusController(
        5, roles, {"leg_front": {"current": 0.4}}, dt_ms=1.0
    )
    assert controller.trigger("leg_front", duration_ms=2) == 2
    first = controller.encode()
    second = controller.encode()
    third = controller.encode()
    assert np.allclose(first, [0.0, 0.4, 0.0, 0.4, 0.0])
    assert np.array_equal(first, second)
    assert not np.any(third)


def test_rear_vision_and_threat_collision_boundary() -> None:
    vision = VisionSensor()
    vision.sense(0, 0, 0, -100, 0, 5, 0.01)
    rear = vision.sense(0, 0, 0, -90, 0, 6, 0.01)
    assert rear.looming > 0
    fly = FlyBody(50, 50)
    world = World(100, 100, threat_speed=100)
    world.threat.x, world.threat.y, world.threat.radius = 51, 50, 10
    world.update(fly, 0.1)
    assert np.hypot(world.threat.x - fly.x, world.threat.y - fly.y) >= world.threat.radius + 12 - 1e-6


def test_array_metadata_is_printable() -> None:
    assert metadata_value(np.array(["a", "b"])) == ["a", "b"]
    assert metadata_value(np.array([])) is None


def test_motor_decoder_uses_spike_window() -> None:
    graph = Connectome.demo()
    decoder = MotorDecoder(graph.roles, window_steps=4, gain=1)
    quiet = decoder.update(np.zeros(graph.neuron_count, dtype=bool))
    spikes = np.zeros(graph.neuron_count, dtype=bool)
    spikes[graph.roles["turn_left"]] = True
    active = decoder.update(spikes)
    assert quiet.turn_left == 0
    assert active.turn_left > active.turn_right


def test_feather_preprocessing_and_body_mapping(tmp_path: Path) -> None:
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    annotations = pd.DataFrame({
        "body": [101, 202, 303],
        "type": ["LPLC2_L", "interneuron", "DNp01_R"],
        "class": ["visual", "central brain", "descending"],
    })
    nts = pd.DataFrame({"body": [101, 202, 303], "consensus_nt": ["acetylcholine"] * 3})
    edges = pd.DataFrame({"body_pre": [101, 202], "body_post": [202, 303], "weight": [10, 12]})
    feather.write_feather(annotations, raw / "annotations.feather")
    feather.write_feather(nts, raw / "neurotransmitters.feather")
    feather.write_feather(edges, raw / "connectome-weights.feather")
    summary = preprocess(raw, processed, config_dir=Path(__file__).parents[1] / "config")
    graph = Connectome.load(processed, mmap_mode=None)
    assert summary["neurons"] == 3 and summary["edges"] == 2
    assert graph.index_for_body(202) == 1
    assert graph.outgoing(0)[0].tolist() == [1]
    assert graph.roles["looming"].tolist() == [0]


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    config = load_config()
    _, brain, _, _, world, fly = build_runtime(config, tmp_path / "missing", True)
    brain.state.voltage[2] = 0.42
    brain.step_number = 17
    fly.x, world.threat.radius = 123.0, 44.0
    path = tmp_path / "snapshot.npz"
    save_snapshot(path, brain, fly, world)
    brain.state.voltage[:] = 0
    brain.step_number = 0
    fly.x, world.threat.radius = 0.0, 0.0
    load_snapshot(path, brain, fly, world)
    assert brain.state.voltage[2] == np.float32(0.42)
    assert brain.step_number == 17 and fly.x == 123.0 and world.threat.radius == 44.0


def test_experiment_log_replay_data(tmp_path: Path) -> None:
    log = ExperimentLog(tmp_path / "run", {"seed": 7, "config": {"dt": 1}})
    log.append({"time_ms": 1.0, "spike_count": 3})
    log.save()
    manifest, frame = ExperimentLog.load(tmp_path / "run")
    assert manifest["seed"] == 7
    assert frame.to_dict("records") == [{"time_ms": 1.0, "spike_count": 3}]


def test_demo_closed_loop_moves_only_after_neural_activity(tmp_path: Path) -> None:
    config = load_config()
    graph, brain, encoder, decoder, world, fly = build_runtime(config, tmp_path / "missing", True)
    initial = (fly.x, fly.y)
    for _ in range(250):
        result = brain.step(encoder.encode_looming(1.0))
        command = decoder.update(result.spikes)
        fly.update(command, 0.001, world.width, world.height)
    assert np.hypot(fly.x - initial[0], fly.y - initial[1]) > 0


def test_deterministic_demo() -> None:
    graph = Connectome.demo()
    first = BrainSimulator(graph)
    second = BrainSimulator(graph)
    encoder = StimulusEncoder(graph.neuron_count, graph.roles["looming"])
    for _ in range(25):
        current = encoder.encode_looming(0.8)
        a, b = first.step(current), second.step(current)
        assert np.array_equal(a.spikes, b.spikes)
    assert np.array_equal(first.state.voltage, second.state.voltage)
