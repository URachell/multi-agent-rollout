import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment specific
    torch = None

from python import (
    Action,
    BatteryAwareRolloutPolicy,
    MultiAgentRolloutEnv,
    RandomMaskedPolicy,
    WarehouseEnvironment,
    run_episode,
    simulate_rollout,
)


class WarehouseEnvironmentTest(unittest.TestCase):
    def test_default_charging_stations_are_created(self):
        env = WarehouseEnvironment(agent_count=2)
        self.assertEqual(len(env.charging_stations), 6)
        self.assertIn((2, 2), env.charging_stations)

    def test_agent_recharges_on_station(self):
        env = WarehouseEnvironment(agent_count=1, battery_capacity=5, move_discharge=2, charge_rate=3)
        env.agent_states[0].position = (2, 2)
        env.agent_states[0].battery = 1

        _, info = env.step([Action.STAY], [env.agent_states[0].position])

        self.assertEqual(info.batteries[0], 4)

    def test_agent_cannot_move_without_battery(self):
        env = WarehouseEnvironment(agent_count=1, battery_capacity=1, move_discharge=2)
        start = env.agent_states[0].position

        _, info = env.step([Action.RIGHT], [start])

        self.assertEqual(env.agent_states[0].position, start)
        self.assertEqual(info.out_of_battery, [0])

    def test_battery_feasible_actions_prioritize_reaching_charger(self):
        env = WarehouseEnvironment(agent_count=1, battery_capacity=6, move_discharge=2, charge_rate=4, low_battery_threshold=6)
        env.agent_states[0].position = (2, 4)
        env.agent_states[0].battery = 6
        feasible = env.battery_feasible_actions(0, (10, 10))
        self.assertIn(Action.LEFT, feasible)
        self.assertNotIn(Action.RIGHT, feasible)

    def test_nearest_charger_returns_current_station(self):
        env = WarehouseEnvironment(agent_count=1)
        charger = (2, 2)
        self.assertEqual(env.nearest_charging_station(charger), charger)

    def test_battery_feasible_actions_do_not_assume_free_recharge_off_station(self):
        env = WarehouseEnvironment(
            agent_count=1,
            battery_capacity=10,
            move_discharge=1,
            idle_discharge=1,
            charge_rate=1,
            low_battery_threshold=10,
            charge_safety_margin=3,
        )
        env.agent_states[0].position = (2, 4)
        env.agent_states[0].battery = 5
        feasible = env.battery_feasible_actions(0, (10, 10))
        self.assertNotIn(Action.STAY, feasible)
        self.assertIn(Action.LEFT, feasible)

    def test_battery_feasible_actions_allow_waiting_on_charger(self):
        env = WarehouseEnvironment(
            agent_count=1,
            battery_capacity=10,
            move_discharge=1,
            idle_discharge=0,
            charge_rate=2,
            low_battery_threshold=10,
            charge_safety_margin=3,
        )
        env.agent_states[0].position = (2, 2)
        env.agent_states[0].battery = 1
        feasible = env.battery_feasible_actions(0, (10, 10))
        self.assertIn(Action.STAY, feasible)
        self.assertEqual(env.nearest_charging_station(env.agent_states[0].position), (2, 2))

    def test_rollout_policy_redirects_low_battery_agent_to_charger(self):
        env = WarehouseEnvironment(agent_count=1, battery_capacity=20, move_discharge=1, charge_rate=5, low_battery_threshold=6)
        env.agent_states[0].position = (2, 4)
        env.agent_states[0].battery = 6
        policy = BatteryAwareRolloutPolicy(seed=0)
        action = policy.choose_action(env, 0, [(10, 10)])
        self.assertEqual(action, Action.LEFT)

    def test_rl_wrapper_exposes_battery_observation_action_mask_and_flat_features(self):
        rl_env = MultiAgentRolloutEnv(num_agents=2, battery_capacity=8, move_discharge=2, charge_rate=4)
        observation = rl_env.reset()
        self.assertIn("agent_batteries", observation)
        masks = rl_env.sample_action_mask()
        self.assertEqual(len(masks), 2)
        self.assertTrue(all(isinstance(mask, list) for mask in masks))
        features = rl_env.flatten_observation(observation)
        self.assertEqual(len(features), rl_env.observation_dim)

    def test_simulate_rollout_returns_transitions(self):
        transitions = simulate_rollout(num_agents=2, max_steps=5, seed=1, battery_capacity=8)
        self.assertGreater(len(transitions), 0)
        self.assertIn("agent_batteries", transitions[0].observation)

    def test_run_episode_reports_completion_metrics(self):
        env = MultiAgentRolloutEnv(num_agents=2, battery_capacity=12, move_discharge=1, charge_rate=4)
        policy = RandomMaskedPolicy(seed=0)
        metrics = run_episode(env, lambda runtime_env: policy.choose_actions(runtime_env.env, runtime_env.targets), max_steps=3)
        self.assertIn("completion_time", metrics)
        self.assertIn("boxes_left", metrics)

    def test_experiment_report_script_renders_markdown_table(self):
        metrics = {
            "battery_rollout": {
                "completion_rate": 1.0,
                "avg_completion_time": 10.0,
                "avg_episode_reward": 50.0,
                "avg_boxes_left": 0.0,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "comparison_metrics.json"
            output_path = Path(tmpdir) / "report.md"
            metrics_path.write_text(json.dumps(metrics))
            import subprocess

            subprocess.run(
                [
                    "python",
                    "scripts/experiment_report.py",
                    str(metrics_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
            )
            report = output_path.read_text()
            self.assertIn("battery_rollout", report)
            self.assertIn("Completion Rate", report)

    def test_visualize_script_can_export_ascii_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frames.txt"
            import subprocess

            subprocess.run(
                [
                    "python",
                    "scripts/visualize_rollout.py",
                    "--policy",
                    "battery_rollout",
                    "--num-agents",
                    "2",
                    "--max-steps",
                    "2",
                    "--save-frames",
                    str(output_path),
                ],
                check=True,
            )
            text = output_path.read_text()
            self.assertIn("step=0", text)
            self.assertIn("FRAME", text)

    def test_repro_suite_dry_run_prints_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.json"
            suite_path.write_text(json.dumps({"experiments": [{"name": "demo", "config": "configs/ppo_smoke.json"}]}))
            import subprocess

            completed = subprocess.run(
                [
                    "python",
                    "scripts/run_experiment_suite.py",
                    str(suite_path),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("scripts/train_rl.py --config configs/ppo_smoke.json", completed.stdout)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_masked_max_q_values_respects_feasible_joint_actions(self):
        from scripts.train_rl import masked_max_q_values

        q_values = torch.tensor([[1.0, 100.0, 3.0], [5.0, 4.0, 3.0]])
        joint_action_mask = torch.tensor([[True, False, True], [False, False, False]])

        next_q = masked_max_q_values(q_values, joint_action_mask)

        self.assertTrue(torch.equal(next_q, torch.tensor([3.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
