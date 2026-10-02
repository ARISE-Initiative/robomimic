"""Bellman n-step returns stop at the first terminal transition."""
import unittest
from types import SimpleNamespace
import torch
from robomimic.algo.bcq import BCQ
from robomimic.algo.cql import CQL
from robomimic.algo.td3_bc import TD3_BC


class NStepTerminalReturnTest(unittest.TestCase):
    def process(self, algorithm, rewards, dones, gamma, infinite=False):
        n_step = rewards.shape[1]
        owner = SimpleNamespace(
            n_step=n_step,
            device=torch.device("cpu"),
            algo_config=SimpleNamespace(n_step=n_step, discount=gamma, infinite_horizon=infinite),
        )
        owner.set_discount = lambda value: setattr(owner, "discount", value)
        batch = {
            "rewards": rewards, "dones": dones,
            "obs": {"state": torch.zeros(rewards.shape[0], n_step, 2)},
            "next_obs": {"state": torch.ones(rewards.shape[0], n_step, 2)},
            "actions": torch.zeros(rewards.shape[0], n_step, 2),
        }
        return algorithm.process_batch_for_training(owner, batch)

    def oracle(self, rewards, dones, gamma, infinite=False):
        totals = []
        for row_rewards, row_dones in zip(rewards.tolist(), dones.tolist()):
            total = 0.
            for step, (reward, done) in enumerate(zip(row_rewards, row_dones)):
                total += gamma ** step * reward / (1 - gamma if infinite and done else 1)
                if done:
                    break
            totals.append([total])
        return torch.tensor(totals, dtype=torch.float32)

    def test_terminal_reward_is_included_once(self):
        rewards = torch.tensor([[2., 2., 2., 2.], [1., 3., 100., 200.],
                                [1., 2., 3., 4.], [1., 2., 3., 4.]])
        dones = torch.tensor([[1., 1., 1., 1.], [0., 1., 0., 0.],
                              [0., 0., 0., 1.], [0., 0., 0., 0.]])
        for algorithm in (BCQ, CQL, TD3_BC):
            for gamma in (0., 0.5, 0.99):
                with self.subTest(algorithm=algorithm.__name__, gamma=gamma):
                    result = self.process(algorithm, rewards, dones, gamma)
                    torch.testing.assert_close(result["rewards"], self.oracle(rewards, dones, gamma))
                    torch.testing.assert_close(result["dones"], torch.tensor([[1.], [1.], [1.], [0.]]))

    def test_absorbing_tail_scales_only_the_terminal_reward(self):
        rewards = torch.tensor([[2., 2., 2., 2.], [1., 3., 100., 200.],
                                [1., 2., 3., 4.], [1., 2., 3., 4.]])
        dones = torch.tensor([[1., 1., 1., 1.], [0., 1., 0., 0.],
                              [0., 0., 0., 1.], [0., 0., 0., 0.]])
        for algorithm in (BCQ, TD3_BC):
            for gamma in (0., 0.5, 0.99):
                with self.subTest(algorithm=algorithm.__name__, gamma=gamma):
                    result = self.process(algorithm, rewards, dones, gamma, infinite=True)
                    torch.testing.assert_close(result["rewards"], self.oracle(rewards, dones, gamma, True))

    def test_nonterminal_and_single_step_returns_are_preserved(self):
        for n_step in (1, 2, 5):
            rewards = torch.arange(1., 1. + 2 * n_step).reshape(2, n_step)
            dones = torch.zeros_like(rewards)
            for algorithm in (BCQ, CQL, TD3_BC):
                with self.subTest(n_step=n_step, algorithm=algorithm.__name__):
                    result = self.process(algorithm, rewards, dones, 0.9)
                    torch.testing.assert_close(result["rewards"], self.oracle(rewards, dones, 0.9))

    def test_integer_rewards_preserve_fractional_discounts(self):
        for dtype in (torch.int64, torch.uint8):
            rewards = torch.tensor([[2, 6, 7, 8], [2, 6, 7, 8]], dtype=dtype)
            dones = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=dtype)
            for algorithm in (BCQ, CQL, TD3_BC):
                for gamma in (0.5, 0.99):
                    for infinite in ((False,) if algorithm is CQL else (False, True)):
                        with self.subTest(dtype=dtype, algorithm=algorithm.__name__,
                                          gamma=gamma, infinite=infinite):
                            result = self.process(algorithm, rewards, dones, gamma, infinite)
                            torch.testing.assert_close(
                                result["rewards"], self.oracle(rewards, dones, gamma, infinite)
                            )
                            self.assertEqual(result["rewards"].dtype, torch.float32)

    def test_rewards_after_termination_have_no_gradient(self):
        for algorithm in (BCQ, CQL, TD3_BC):
            with self.subTest(algorithm=algorithm.__name__):
                rewards = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
                result = self.process(algorithm, rewards, torch.tensor([[0., 1., 1., 1.]]), 0.5)
                result["rewards"].sum().backward()
                torch.testing.assert_close(rewards.grad, torch.tensor([[1., 0.5, 0., 0.]]))


if __name__ == "__main__":
    unittest.main()

