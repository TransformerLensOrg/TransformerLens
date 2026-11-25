import torch

from transformer_lens.utilities import print_gpu_mem


def test_print_gpu_mem_reports_gib(monkeypatch, capsys):
    """The allocated byte count must be converted with 2**30 bytes per GiB."""
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 3 * 2**30)

    print_gpu_mem("step")

    assert capsys.readouterr().out.strip() == "step ~ 3.0 GiB allocated on GPU."


def test_print_gpu_mem_reports_fractional_gib(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 2**29)

    print_gpu_mem()

    assert "0.5 GiB allocated on GPU." in capsys.readouterr().out
