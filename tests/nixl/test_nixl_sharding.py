import unittest
from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.nixl_spec import NIXLSharding

pytestmark = pytest.mark.cpu_test


class TestNIXLSharding(unittest.TestCase):
    """Test cases for NIXLSharding class"""

    def setUp(self):
        """Set up test fixtures"""
        pass

    def test_default_sharding(self):
        """Test default sharding creation"""
        sharding = NIXLSharding.default()
        self.assertEqual(sharding.shard_mesh, OrderedDict([(0, 1)]))
        self.assertEqual(sharding.shard_indices, [(0,)])

    def test_basic_sharding_creation(self):
        """Test basic sharding creation with 2D sharding"""
        # Test the example from comments: {0: 2, 1: 8} for 2D sharding
        shard_mesh = OrderedDict([(0, 2), (1, 8)])
        shard_indices = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
        sharding = NIXLSharding(shard_mesh=shard_mesh, shard_indices=shard_indices)

        self.assertEqual(sharding.shard_mesh, shard_mesh)
        self.assertEqual(sharding.shard_indices, shard_indices)

    def test_validation_invalid_tuple_lengths(self):
        """Test validation with tuples of different lengths"""
        shard_mesh = OrderedDict([(0, 2), (1, 2)])
        shard_indices = [
            (0, 0),
            (0, 1),
            (1,),
        ]  # Invalid: last tuple has different length

        with self.assertRaises(ValueError) as context:
            NIXLSharding(shard_mesh=shard_mesh, shard_indices=shard_indices)

        self.assertIn("All tuples must have the same length", str(context.exception))

    def test_validation_non_increasing_order(self):
        """Test validation with non-increasing shard indices"""
        shard_mesh = OrderedDict([(0, 2), (1, 2)])
        shard_indices = [(0, 0), (0, 1), (0, 0)]  # Invalid: not strictly increasing

        with self.assertRaises(ValueError) as context:
            NIXLSharding(shard_mesh=shard_mesh, shard_indices=shard_indices)

        self.assertIn("Shard indices must be in strictly increasing order", str(context.exception))

    def test_find_finest_shard_mesh_basic(self):
        """Test finding finest shard_mesh from multiple shardings"""
        # Create multiple shardings with different granularities
        sharding1 = NIXLSharding(
            shard_mesh=OrderedDict([(0, 2), (1, 4)]),
            shard_indices=[
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
            ],
        )

        sharding2 = NIXLSharding(
            shard_mesh=OrderedDict([(0, 4), (1, 2)]),
            shard_indices=[
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
            ],
        )

        sharding3 = NIXLSharding(
            shard_mesh=OrderedDict([(1, 8)]),
            shard_indices=[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
        )

        sharding_list = [sharding1, sharding2, sharding3]
        # API change: find_finest_shard_mesh now takes list[OrderedDict], not list[NIXLSharding]
        shard_mesh_list = [s.shard_mesh for s in sharding_list]
        finest_shard_mesh = NIXLSharding.find_finest_shard_mesh(shard_mesh_list)

        # The finest mesh uses dimension-wise LCM values of 4 and 8.
        expected = OrderedDict([(0, 4), (1, 8)])
        self.assertEqual(finest_shard_mesh, expected)

    def test_find_finest_shard_mesh_single_dimension(self):
        """Test finding finest shard_mesh with single dimension shardings"""
        sharding1 = NIXLSharding(shard_mesh=OrderedDict([(0, 3)]), shard_indices=[(0,), (1,), (2,)])

        sharding2 = NIXLSharding(
            shard_mesh=OrderedDict([(0, 5)]),
            shard_indices=[(0,), (1,), (2,), (3,), (4,)],
        )

        sharding_list = [sharding1, sharding2]
        # API change: find_finest_shard_mesh now takes list[OrderedDict], not list[NIXLSharding]
        shard_mesh_list = [s.shard_mesh for s in sharding_list]
        finest_shard_mesh = NIXLSharding.find_finest_shard_mesh(shard_mesh_list)

        # Expected: LCM(3, 5) = 15
        expected = OrderedDict([(0, 15)])
        self.assertEqual(finest_shard_mesh, expected)

    def test_refactor_based_on_finer_shard_mesh_basic(self):
        """Test basic refactoring to finer sharding"""
        # Refinement expands the original shard into four positions on the finer mesh.

        current_sharding = NIXLSharding(
            shard_mesh=OrderedDict([(1, 2)]),
            shard_indices=[(1,)],  # 2nd shard in dimension 1
        )

        finer_shard_mesh = OrderedDict([(0, 2), (1, 4)])

        current_sharding.refactor_based_on_finer_shard_mesh(finer_shard_mesh)

        # Verify the refactored sharding
        self.assertEqual(current_sharding.shard_mesh, finer_shard_mesh)
        expected_shard_indices = [(0, 2), (0, 3), (1, 2), (1, 3)]
        self.assertEqual(current_sharding.shard_indices, expected_shard_indices)

    def test_refactor_based_on_finer_shard_mesh_expansion(self):
        """Test refactoring with dimension expansion"""
        # Refinement expands dimension 0 and adds every index from dimension 1.

        current_sharding = NIXLSharding(
            shard_mesh=OrderedDict([(0, 2)]),
            shard_indices=[(1,)],  # 2nd shard in dimension 0
        )

        finer_shard_mesh = OrderedDict([(0, 4), (1, 3)])

        current_sharding.refactor_based_on_finer_shard_mesh(finer_shard_mesh)

        # Verify the refactored sharding
        self.assertEqual(current_sharding.shard_mesh, finer_shard_mesh)
        # Expected: shard 1 in dim 0 becomes shards 2,3 in finer dim 0
        # And each gets expanded by dim 1 with indices 0,1,2
        expected_shard_indices = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.assertEqual(current_sharding.shard_indices, expected_shard_indices)

    def test_refactor_incompatible_sharding(self):
        """Test refactoring with incompatible sharding"""
        current_sharding = NIXLSharding(shard_mesh=OrderedDict([(0, 3)]), shard_indices=[(0,), (1,), (2,)])

        # Incompatible: 4 is not divisible by 3
        finer_shard_mesh = OrderedDict([(0, 4)])

        with self.assertRaises(ValueError) as context:
            current_sharding.refactor_based_on_finer_shard_mesh(finer_shard_mesh)

        self.assertIn("is not divisible by current shard_mesh", str(context.exception))

    def test_get_local_sharded_tensors_multiple_shards(self):
        """Test getting local sharded tensors with multiple shards"""
        # Create a 2D tensor: 4x6
        tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, :3].contiguous()

        # Sharding: split dim 0 into 2 shards, dim 1 into 2 shards
        # Take the first shard from dimension 0 and the second from dimension 1.
        sharding = NIXLSharding(
            shard_mesh=OrderedDict([(1, 2), (0, 2)]),
            shard_indices=[(0, 0), (0, 1)],  # all rows, 0-2 column
        )
        self.assertEqual(sharding._local_shard_mesh, OrderedDict([(1, 1), (0, 2)]))
        local_shards = sharding.get_local_sharded_tensors(tensor)

        # Should get four shards
        self.assertEqual(len(local_shards), 2)

        # First shard: rows 0-1, cols 0-2
        expected1 = torch.tensor([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]], dtype=torch.float32)
        # Second shard: rows 2-3, cols 0-2
        expected2 = torch.tensor([[12.0, 13.0, 14.0], [18.0, 19.0, 20.0]], dtype=torch.float32)

        torch.testing.assert_close(local_shards[0], expected1)
        torch.testing.assert_close(local_shards[1], expected2)

    def test_complex_refactor_scenario(self):
        """Test a complex refactoring scenario with multiple dimensions"""
        # Start with a simple 1D sharding
        current_sharding = NIXLSharding(shard_mesh=OrderedDict([(0, 2)]), shard_indices=[(0,), (1,)])

        # Refactor to a complex 3D sharding
        finer_shard_mesh = OrderedDict([(0, 4), (1, 3), (2, 2)])

        current_sharding.refactor_based_on_finer_shard_mesh(finer_shard_mesh)

        # Verify the result
        self.assertEqual(current_sharding.shard_mesh, finer_shard_mesh)

        # Should have 4 * 3 * 2 = 24 shard indices
        self.assertEqual(len(current_sharding.shard_indices), 24)

        # Verify all indices are unique and in order
        self.assertEqual(len(set(current_sharding.shard_indices)), 24)
        for i in range(1, len(current_sharding.shard_indices)):
            self.assertLess(current_sharding.shard_indices[i - 1], current_sharding.shard_indices[i])


if __name__ == "__main__":
    unittest.main()
