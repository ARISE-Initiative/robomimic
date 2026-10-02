"""The weighted sampler must honor its supplied torch.Generator."""
import unittest
import numpy as np
import torch
from robomimic.utils.dataset import CustomWeightedRandomSampler


class WeightedSamplerGeneratorTest(unittest.TestCase):
    def make_sampler(self, generator, replacement=True, count=100):
        return CustomWeightedRandomSampler(
            weights=[1., 2., 3., 4.], num_samples=count,
            replacement=replacement, generator=generator,
        )

    def test_same_generator_state_is_independent_of_numpy_seed(self):
        outputs = []
        for numpy_seed in (1, 200):
            np.random.seed(numpy_seed)
            generator = torch.Generator().manual_seed(9876)
            outputs.append(list(self.make_sampler(generator)))
        self.assertEqual(outputs[0], outputs[1])

    def test_generator_advances_and_restoring_it_replays_the_next_iterator(self):
        generator = torch.Generator().manual_seed(1234)
        initial_state = generator.get_state().clone()
        sampler = self.make_sampler(generator)
        first = list(sampler)
        self.assertFalse(torch.equal(initial_state, generator.get_state()))
        saved_state = generator.get_state().clone()
        second = list(sampler)
        self.assertNotEqual(first, second)
        generator.set_state(saved_state)
        np.random.seed(42)
        self.assertEqual(second, list(sampler))

    def test_weighted_probabilities_and_zero_weight_categories(self):
        sampler = CustomWeightedRandomSampler(
            [0., 1., 3.], 50000, replacement=True,
            generator=torch.Generator().manual_seed(17),
        )
        samples = np.array(list(sampler))
        self.assertFalse(np.any(samples == 0))
        np.testing.assert_allclose(np.bincount(samples, minlength=3) / len(samples),
                                   [0., 0.25, 0.75], atol=0.01)

    def test_sampling_without_replacement_preserves_unique_indices(self):
        sampler = self.make_sampler(torch.Generator().manual_seed(19),
                                    replacement=False, count=4)
        self.assertEqual(sorted(sampler), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

