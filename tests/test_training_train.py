"""Tests for training functionality."""

import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.training.train import (
    _run_training_epoch,
    _setup_amp,
    _setup_data_and_optimizer,
    _setup_model,
    _setup_training_parameters,
    main,
    train_finetune,
)


class TestSetupTrainingParameters:
    """Test _setup_training_parameters function."""

    @mock.patch("src.training.train.detect_covers_root")
    @mock.patch("os.path.isabs")
    @mock.patch("os.path.join")
    def test_setup_training_parameters_defaults(self, mock_join, mock_isabs, mock_detect_root):
        """Test setting up training parameters with defaults."""
        mock_detect_root.return_value = "/covers/root"
        mock_isabs.return_value = False
        mock_join.side_effect = lambda *args: "/".join(args)

        with mock.patch("src.training.train.config") as mock_config:
            mock_config.training.jsonl_path = "metadata.jsonl"
            mock_config.training.epochs = 5
            mock_config.training.batch_size = 32
            mock_config.training.lr = 5e-4
            mock_config.training.workers = 8
            mock_config.training.dim = 512
            mock_config.training.amp = "fp16"
            mock_config.training.out_path = "cover_encoder.pt"
            mock_config.training.grad_accumulation_steps = 4

            result = _setup_training_parameters()

            assert len(result) == 12
            jsonl_path, root_dir, epochs, batch_size, lr, workers, dim, amp, out_path, device, grad_accumulation_steps, effective_batch_size = result[:12]

            assert jsonl_path == "/covers/root/metadata.jsonl"
            assert root_dir == "/covers/root"
            assert epochs == 5
            assert batch_size == 32
            assert lr == 5e-4
            assert workers == 8
            assert dim == 512
            assert amp == "fp16"
            assert out_path == "/covers/root/cover_encoder.pt"
            assert grad_accumulation_steps == 4
            assert effective_batch_size == 128  # batch_size * grad_accumulation_steps

    def test_setup_training_parameters_custom_values(self):
        """Test setting up training parameters with custom values."""
        with mock.patch("src.training.train.detect_covers_root", return_value="/custom/root"):
            with mock.patch("os.path.isabs", return_value=True):
                result = _setup_training_parameters(
                    jsonl_path="/custom/data.jsonl",
                    root_dir="/custom/root",
                    epochs=10,
                    batch_size=16,
                    lr=1e-4,
                    workers=4,
                    dim=256,
                    amp="bf16",
                    out_path="/custom/model.pt",
                    device="cuda",
                    grad_accumulation_steps=2,
                )

                assert result[0] == "/custom/data.jsonl"  # jsonl_path
                assert result[1] == "/custom/root"        # root_dir
                assert result[2] == 10                    # epochs
                assert result[3] == 16                    # batch_size
                assert result[4] == 1e-4                  # lr
                assert result[5] == 4                     # workers
                assert result[6] == 256                   # dim
                assert result[7] == "bf16"                # amp
                assert result[8] == "/custom/model.pt"    # out_path
                assert result[9] == "cuda"                # device
                assert result[10] == 2                    # grad_accumulation_steps
                assert result[11] == 32                   # effective_batch_size

    def test_setup_training_parameters_invalid_grad_accumulation(self):
        """Test that invalid grad_accumulation_steps raises error."""
        with pytest.raises(ValueError, match="grad_accumulation_steps must be >= 1"):
            _setup_training_parameters(grad_accumulation_steps=0)


class TestSetupModel:
    """Test _setup_model function."""

    @mock.patch("open_clip.create_model_and_transforms")
    def test_setup_model_with_projection(self, mock_create_model):
        """Test setting up model when projection is needed."""
        mock_base = mock.MagicMock()
        mock_base.embed_dim = 768
        mock_preprocess = mock.MagicMock()

        mock_create_model.return_value = (mock_base, None, mock_preprocess)

        with mock.patch("src.training.train.config") as mock_config:
            mock_config.model.clip_model = "ViT-B-16"
            mock_config.model.clip_pretrained = "laion2b_s34b_b88k"

            model = _setup_model("cuda", dim=512)

            assert model is not None
            mock_create_model.assert_called_once_with("ViT-B-16", pretrained="laion2b_s34b_b88k")

            # Check that model was moved to device
            # Note: We can't easily check the internal calls, but we can check the model exists

    @mock.patch("open_clip.create_model_and_transforms")
    def test_setup_model_no_projection(self, mock_create_model):
        """Test setting up model when no projection is needed."""
        mock_base = mock.MagicMock()
        mock_base.embed_dim = 512  # Same as target dim
        mock_preprocess = mock.MagicMock()

        mock_create_model.return_value = (mock_base, None, mock_preprocess)

        model = _setup_model("cpu", dim=512)

        assert model is not None


class TestSetupDataAndOptimizer:
    """Test _setup_data_and_optimizer function."""

    @mock.patch("src.training.train.CoverAugDataset")
    @mock.patch("torch.utils.data.DataLoader")
    @mock.patch("torch.optim.AdamW")
    @mock.patch("torch.optim.lr_scheduler.CosineAnnealingLR")
    def test_setup_data_and_optimizer(self, mock_scheduler, mock_optimizer, mock_dataloader, mock_dataset):
        """Test setting up data, optimizer and scheduler."""
        # Mock dataset
        mock_dataset_instance = mock.MagicMock()
        mock_dataset_instance.__len__ = mock.MagicMock(return_value=10)
        mock_dataset.return_value = mock_dataset_instance

        # Mock dataloader
        mock_dl_instance = mock.MagicMock()
        mock_dl_instance.__len__ = mock.MagicMock(return_value=10)
        mock_dataloader.return_value = mock_dl_instance

        # Mock optimizer
        mock_opt_instance = mock.MagicMock()
        mock_optimizer.return_value = mock_opt_instance

        # Mock scheduler
        mock_sched_instance = mock.MagicMock()
        mock_scheduler.return_value = mock_sched_instance

        # Mock model
        mock_model = mock.MagicMock()

        dl, opt, sched = _setup_data_and_optimizer(
            jsonl_path="data.jsonl",
            root_dir="/data",
            batch_size=16,
            workers=4,
            lr=1e-4,
            grad_accumulation_steps=2,
            epochs=5,
            net=mock_model,
        )

        assert opt == mock_opt_instance
        assert sched == mock_sched_instance

        # Verify calls
        mock_dataset.assert_called_once_with("data.jsonl", "/data")
        mock_optimizer.assert_called_once_with(mock_model.parameters(), lr=1e-4, weight_decay=1e-4)
        mock_scheduler.assert_called_once()


class TestSetupAmp:
    """Test _setup_amp function."""

    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_setup_amp_fp16_cuda(self, mock_cuda_available):
        """Test AMP setup for fp16 on CUDA."""
        device_type, amp_dtype, use_scaler, scaler = _setup_amp("fp16")

        assert device_type == "cuda"
        assert amp_dtype == torch.float16
        assert use_scaler is True
        assert scaler is not None

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_setup_amp_fp16_cpu(self, mock_cuda_available):
        """Test AMP setup for fp16 on CPU."""
        device_type, amp_dtype, use_scaler, scaler = _setup_amp("fp16")

        assert device_type == "cpu"
        assert amp_dtype is None  # fp16 not supported on CPU in this setup
        assert use_scaler is False

    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_setup_amp_bf16_cuda(self, mock_cuda_available):
        """Test AMP setup for bf16 on CUDA."""
        device_type, amp_dtype, use_scaler, scaler = _setup_amp("bf16")

        assert device_type == "cuda"
        assert amp_dtype == torch.bfloat16
        assert use_scaler is False

    def test_setup_amp_none(self):
        """Test AMP setup with none."""
        device_type, amp_dtype, use_scaler, scaler = _setup_amp("none")

        assert amp_dtype is None
        assert use_scaler is False


class TestRunTrainingEpoch:
    """Test _run_training_epoch function."""

    @mock.patch("src.training.train.tqdm")
    @mock.patch("torch.no_grad")
    def test_run_training_epoch(self, mock_no_grad, mock_tqdm):
        """Test running a training epoch."""
        # Mock model
        mock_model = mock.MagicMock()
        mock_model.return_value = torch.randn(2, 512)

        # Mock dataloader
        mock_dl = mock.MagicMock()
        mock_batch1 = (torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224))
        mock_batch2 = (torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224))
        mock_dl.__iter__ = mock.MagicMock(return_value=iter([mock_batch1, mock_batch2]))
        mock_dl.__len__ = mock.MagicMock(return_value=2)

        # Mock optimizer and scheduler
        mock_opt = mock.MagicMock()
        mock_sched = mock.MagicMock()

        # Mock loss function
        mock_loss_fn = mock.MagicMock()
        loss_tensor = torch.tensor(1.5, requires_grad=True)
        mock_loss_fn.return_value = loss_tensor

        # Mock tqdm
        mock_pbar = mock.MagicMock()
        mock_pbar.__iter__ = mock.MagicMock(return_value=iter([mock_batch1, mock_batch2]))
        mock_tqdm.return_value = mock_pbar

        _run_training_epoch(
            net=mock_model,
            dl=mock_dl,
            opt=mock_opt,
            sched=mock_sched,
            loss_fn=mock_loss_fn,
            device="cpu",
            amp="none",
            device_type="cpu",
            amp_dtype=None,
            use_scaler=False,
            scaler=None,
            grad_accumulation_steps=1,
            ep=0,
            epochs=5,
        )

        # Verify optimizer and scheduler were called
        assert mock_opt.zero_grad.called
        assert mock_opt.step.called
        assert mock_sched.step.called


class TestTrainFinetune:
    """Test train_finetune function."""

    @mock.patch("src.training.train._setup_training_parameters")
    @mock.patch("src.training.train._setup_model")
    @mock.patch("src.training.train._setup_data_and_optimizer")
    @mock.patch("src.training.train._setup_amp")
    @mock.patch("src.training.train._run_training_epoch")
    @mock.patch("torch.save")
    def test_train_finetune(self, mock_save, mock_run_epoch, mock_setup_amp, mock_setup_data_opt,
                           mock_setup_model, mock_setup_params):
        """Test the main training function."""
        # Setup mocks
        mock_setup_params.return_value = (
            "data.jsonl", "/root", 2, 16, 1e-4, 4, 512, "none", "model.pt", "cpu", 1, 16
        )

        mock_model = mock.MagicMock()
        mock_setup_model.return_value = mock_model

        mock_dl = mock.MagicMock()
        mock_opt = mock.MagicMock()
        mock_sched = mock.MagicMock()
        mock_setup_data_opt.return_value = (mock_dl, mock_opt, mock_sched)

        mock_loss_fn = mock.MagicMock()
        with mock.patch("src.training.train.NTXent", return_value=mock_loss_fn):
            mock_setup_amp.return_value = ("cpu", None, False, None)

            train_finetune()

            # Verify training was called for each epoch
            assert mock_run_epoch.call_count == 2  # 2 epochs

            # Verify model was saved
            mock_save.assert_called_once()


class TestMain:
    """Test main function."""

    @mock.patch("argparse.ArgumentParser")
    @mock.patch("src.training.train.train_finetune")
    def test_main(self, mock_train_finetune, mock_parser_class):
        """Test the CLI main function."""
        # Mock argument parser
        mock_parser = mock.MagicMock()
        mock_args = mock.MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        main()

        # Verify train_finetune was called with parsed args
        mock_train_finetune.assert_called_once()
        args_dict = mock_train_finetune.call_args[1]  # kwargs
        assert args_dict == vars(mock_args)
