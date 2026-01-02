"""Tests for data generation and dataset classes."""

import pytest
import numpy as np
import torch

from src.data import (
    OracleDataPoint,
    DatasetConfig,
    DatasetMetadata,
    OracleGenerator,
    generate_oracle_data,
    encode_infoset,
    get_input_dim,
    CFVDataset,
    CFVDatasetFromConfig,
    save_dataset,
    load_dataset,
    create_dataloaders,
)
from src.games.kuhn import KuhnPoker
from src.games.leduc import LeducPoker


class TestSchema:
    """Test schema dataclasses."""
    
    def test_oracle_datapoint_to_dict(self):
        """Test OracleDataPoint serialization."""
        dp = OracleDataPoint(
            infoset_encoding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            cfv=np.array([0.5, -0.5], dtype=np.float32),
            strategy=np.array([0.6, 0.4], dtype=np.float32),
            regrets=np.array([1.0, -1.0], dtype=np.float32),
            infoset_key="test_key",
            player=0,
            reach_prob=0.5,
            weight=1.0,
        )
        
        d = dp.to_dict()
        assert d['infoset_key'] == "test_key"
        assert d['player'] == 0
        assert d['reach_prob'] == 0.5
        
        # Round-trip
        dp2 = OracleDataPoint.from_dict(d)
        assert dp2.infoset_key == dp.infoset_key
        np.testing.assert_array_almost_equal(dp2.cfv, dp.cfv)
    
    def test_dataset_config_validation(self):
        """Test DatasetConfig validation."""
        # Valid config
        config = DatasetConfig(
            game="kuhn",
            cfr_iterations=100,
            num_samples=None,
        )
        assert config.game == "kuhn"
        
        # Invalid game
        with pytest.raises(ValueError):
            DatasetConfig(game="invalid", cfr_iterations=100, num_samples=None)
        
        # Invalid sampling method
        with pytest.raises(ValueError):
            DatasetConfig(
                game="kuhn",
                cfr_iterations=100,
                num_samples=None,
                sampling_method="invalid",
            )


class TestOracleGenerator:
    """Test oracle data generation."""
    
    def test_generate_kuhn_data(self):
        """Test generating data for Kuhn poker."""
        datapoints, metadata = generate_oracle_data(
            game="kuhn",
            cfr_iterations=500,
            solver_type="cfr+",
            seed=42,
        )
        
        assert len(datapoints) == 12  # Kuhn has 12 infosets
        assert metadata.num_unique_infosets == 12
        assert metadata.input_dim == 9  # 3 (card) + 6 (history)
        assert metadata.num_actions == 2
        assert metadata.final_exploitability < 0.1  # Should be near Nash
        
        # Check datapoint structure
        dp = datapoints[0]
        assert dp.infoset_encoding.shape == (9,)
        assert dp.cfv.shape == (2,)
        assert dp.strategy.shape == (2,)
        assert 0 <= dp.player <= 1
    
    def test_generate_leduc_data(self):
        """Test generating data for Leduc poker."""
        datapoints, metadata = generate_oracle_data(
            game="leduc",
            cfr_iterations=200,  # Fewer for speed
            solver_type="cfr+",
            seed=42,
        )
        
        assert len(datapoints) > 100  # Leduc has ~288 infosets
        assert metadata.input_dim == 32
        assert metadata.num_actions == 3
        
        # Check datapoint structure
        dp = datapoints[0]
        assert dp.infoset_encoding.shape == (32,)
        assert dp.cfv.shape == (3,)
    
    def test_sampling(self):
        """Test that sampling works correctly."""
        datapoints, _ = generate_oracle_data(
            game="kuhn",
            cfr_iterations=100,
            num_samples=5,
            solver_type="vanilla",
            seed=42,
        )
        
        assert len(datapoints) == 5


class TestEncoding:
    """Test infoset encoding functions."""
    
    def test_kuhn_encoding_dimension(self):
        """Test Kuhn encoding has correct dimension."""
        assert get_input_dim("kuhn") == 9
    
    def test_leduc_encoding_dimension(self):
        """Test Leduc encoding has correct dimension."""
        assert get_input_dim("leduc") == 32
    
    def test_kuhn_card_encoding(self):
        """Test that Kuhn cards are encoded correctly."""
        game = KuhnPoker()
        # Create a simple state and check encoding
        state = game.initial_state()
        
        # Get first chance outcome
        outcomes = game.chance_outcomes(state)
        state1, _ = outcomes[0]
        
        infoset = game.get_infoset(state1, 0)
        encoding = encode_infoset(infoset, game)
        
        assert encoding.shape == (9,)
        # Card should be one-hot (exactly one 1 in first 3 elements)
        assert encoding[:3].sum() == 1.0


class TestCFVDataset:
    """Test PyTorch dataset class."""
    
    def test_dataset_creation(self):
        """Test creating a dataset."""
        datapoints, _ = generate_oracle_data(
            game="kuhn",
            cfr_iterations=100,
            solver_type="cfr+",
        )
        
        dataset = CFVDataset(datapoints, normalize_cfv=True)
        
        assert len(dataset) == 12
        assert dataset.input_dim == 9
        assert dataset.num_actions == 2
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        datapoints, _ = generate_oracle_data(
            game="kuhn",
            cfr_iterations=100,
            solver_type="cfr+",
        )
        
        dataset = CFVDataset(datapoints, normalize_cfv=True)
        sample = dataset[0]
        
        assert 'encoding' in sample
        assert 'cfv' in sample
        assert 'strategy' in sample
        assert 'weight' in sample
        
        assert sample['encoding'].dtype == torch.float32
        assert sample['cfv'].shape == (2,)
    
    def test_cfv_normalization(self):
        """Test CFV normalization."""
        datapoints, _ = generate_oracle_data(
            game="kuhn",
            cfr_iterations=100,
            solver_type="cfr+",
        )
        
        # Without normalization
        dataset_raw = CFVDataset(datapoints, normalize_cfv=False)
        
        # With normalization
        dataset_norm = CFVDataset(datapoints, normalize_cfv=True)
        
        # Normalized CFVs should be smaller
        raw_max = max(abs(dataset_raw[i]['cfv']).max().item() for i in range(len(dataset_raw)))
        norm_max = max(abs(dataset_norm[i]['cfv']).max().item() for i in range(len(dataset_norm)))
        
        assert norm_max <= 1.0 or norm_max < raw_max


class TestDataLoaders:
    """Test dataloader creation."""
    
    def test_create_dataloaders(self):
        """Test creating train/val/test dataloaders."""
        datapoints, _ = generate_oracle_data(
            game="kuhn",
            cfr_iterations=100,
            solver_type="cfr+",
        )
        
        dataset = CFVDataset(datapoints)
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=4,
            train_split=0.8,
            val_split=0.1,
        )
        
        # Check we got dataloaders
        assert len(train_loader) > 0
        
        # Check batch structure
        batch = next(iter(train_loader))
        assert 'encoding' in batch
        assert batch['encoding'].ndim == 2  # [batch, features]
    
    def test_dataloader_batch_shapes(self):
        """Test that batch shapes are correct."""
        datapoints, _ = generate_oracle_data(
            game="leduc",
            cfr_iterations=100,
            solver_type="cfr+",
        )
        
        dataset = CFVDataset(datapoints)
        train_loader, _, _ = create_dataloaders(dataset, batch_size=8)
        
        batch = next(iter(train_loader))
        
        # Batch dimension should be <= 8
        assert batch['encoding'].shape[0] <= 8
        assert batch['encoding'].shape[1] == 32  # Leduc input dim
        assert batch['cfv'].shape[1] == 3  # Leduc num actions


class TestSaveLoad:
    """Test save/load functionality."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading a dataset."""
        datapoints, metadata = generate_oracle_data(
            game="kuhn",
            cfr_iterations=100,
            solver_type="cfr+",
        )
        
        save_path = tmp_path / "test_dataset"
        save_dataset(datapoints, metadata, str(save_path))
        
        # Check files were created
        assert (tmp_path / "test_dataset_data.npz").exists()
        assert (tmp_path / "test_dataset_meta.json").exists()
        
        # Load and verify
        loaded_dataset, loaded_metadata = load_dataset(str(save_path))
        
        assert len(loaded_dataset) == len(datapoints)
        assert loaded_metadata.num_unique_infosets == metadata.num_unique_infosets
        assert loaded_metadata.config.game == "kuhn"


class TestIntegration:
    """Integration tests for the full data pipeline."""
    
    def test_full_pipeline_kuhn(self):
        """Test full pipeline from config to dataloader."""
        config = DatasetConfig(
            game="kuhn",
            cfr_iterations=200,
            num_samples=None,
            solver_type="cfr+",
            seed=42,
        )
        
        dataset = CFVDatasetFromConfig(config)
        
        assert len(dataset) == 12
        assert dataset.input_dim == 9
        
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, batch_size=4
        )
        
        # Run one training step
        for batch in train_loader:
            x = batch['encoding']
            y = batch['cfv']
            
            assert x.shape[1] == 9
            assert y.shape[1] == 2
            break
    
    def test_reproducibility(self):
        """Test that same seed gives same data."""
        dp1, _ = generate_oracle_data(game="kuhn", cfr_iterations=100, seed=42)
        dp2, _ = generate_oracle_data(game="kuhn", cfr_iterations=100, seed=42)
        
        # Same infoset keys
        keys1 = sorted([dp.infoset_key for dp in dp1])
        keys2 = sorted([dp.infoset_key for dp in dp2])
        assert keys1 == keys2

