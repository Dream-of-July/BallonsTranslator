"""Offline confidence and compatibility checks; no weights or downloads."""
import copy
import math
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from ballontranslator.modules.ocr.mit48px import Model48pxOCR, OCR, normalized_confidence
from ballontranslator.modules.ocr.ocr_mit import OCRMIT48px
from ballontranslator.utils.textblock import TextBlock


@pytest.fixture
def module(monkeypatch):
    monkeypatch.setattr(OCRMIT48px, 'params', copy.deepcopy(OCRMIT48px.params))
    # Exercise the real module/wrapper constructors, replacing only model IO.
    with mock.patch('ballontranslator.modules.ocr.mit48px.OCR') as network, \
            mock.patch('torch.load', return_value={}), \
            mock.patch('builtins.open', mock.mock_open(read_data='<PAD>\n<S>\n</S>\n<SP>\n甲\n乙\n')):
        network.return_value.dictionary = ['<PAD>', '<S>', '</S>', '<SP>', '甲', '乙']
        result = OCRMIT48px(device='cpu')  # Old saved params omit both new keys.
        result.model = Model48pxOCR('unused', 'cpu')
    return result


def reading(joint, normalized, token=4):
    return (torch.tensor([token, 2]), joint, torch.full((2, 3), .5),
            torch.full((2, 3), .75), torch.tensor([[0., 1.]] * 2),
            torch.tensor([[0., 1.]] * 2), normalized)


def block_with_lines(count, manga=False):
    lines = [[[i * 15, 0], [i * 15 + 12, 0], [i * 15 + 12, 40], [i * 15, 40]]
             for i in range(count)]
    return TextBlock(xyxy=[0, 0, count * 15, 40], lines=lines,
                     manga_area=lines[0] if manga else None)


@pytest.mark.parametrize('length', [1, 10, 100])
def test_normalized_confidence_removes_length_bias(length):
    log_probs = torch.full((length,), math.log(.85))
    assert normalized_confidence(log_probs.sum(), torch.full((length,), 4)) == pytest.approx(.85)
    assert normalized_confidence(torch.tensor(math.log(.08)), torch.tensor([4])) < .2


def test_eos_padding_and_bos_are_not_counted():
    # BOS is excluded by the caller; only the first EOS carries probability.
    assert normalized_confidence(torch.tensor(2 * math.log(.8)), torch.tensor([4, 2, 2, 2])) == pytest.approx(.8)
    assert normalized_confidence(torch.tensor(0.), torch.empty(0, dtype=torch.long)) == 0


@pytest.mark.parametrize('headless', [False, True])
def test_default_output_bytes_and_colors_match_legacy(module, monkeypatch, headless):
    from ballontranslator.utils import shared
    monkeypatch.setattr(shared, 'HEADLESS', headless)
    block = block_with_lines(4)
    module.model.model.infer_beam_batch_tensor.return_value = [
        reading(.199, .99), reading(.2, .4), reading(.9, .6, 5), reading(.85, .85)]
    assert module.get_param_value('confidence mode') == 'joint'
    assert module.get_param_value('min confidence') == .2
    module.run_ocr(np.full((48, 64, 3), 255, np.uint8), [block])
    assert block.get_text().encode('utf8') == '甲乙甲'.encode('utf8')
    # Existing color total arithmetic uses the full detected line count.
    np.testing.assert_array_equal(block.fg_colors, [95.25] * 3)
    np.testing.assert_array_equal(block.bg_colors, [143] * 3)
    assert block.manga_line_confidence == []


def test_normalized_param_is_opt_in(module):
    block = block_with_lines(2)
    module.set_param_value('confidence mode', 'normalized')
    module.set_param_value('min confidence', .7)
    module.model.model.infer_beam_batch_tensor.return_value = [reading(.01, .85), reading(.3, .08, 5)]
    module.run_ocr(np.full((48, 48, 3), 255, np.uint8), [block])
    assert block.get_text() == '甲'


@pytest.mark.parametrize('values,expected,partial', [([.85, .1], '甲', True), ([.1, .1], '', False)])
def test_manga_gates_lines_without_saved_param_changes(module, values, expected, partial):
    from ballontranslator.engine.manga_quality import source_review
    block = block_with_lines(2, manga=True)
    block.manga_ocr_confidence = .01  # Legacy metadata cannot veto survivors.
    saved = copy.deepcopy(module.params)
    module.model.model.infer_beam_batch_tensor.return_value = [reading(.01, v) for v in values]
    image = np.full((48, 48, 3), 255, np.uint8)
    module.run_ocr(image, [block])
    assert block.get_text() == expected
    assert block.manga_line_confidence == values
    assert block.manga_partial is partial
    assert source_review(block) == (None if expected else 'unreadable')
    assert module.params == saved
    # A repeated run replaces confidence/line state instead of accumulating it.
    module.run_ocr(image, [block])
    assert block.manga_line_confidence == values
    assert block.manga_line_text == (['甲', ''] if expected else ['', ''])


class SyntheticDecoders:
    def __len__(self):
        return 1

    def __call__(self, embedded, cached, memory, mask, step):
        return embedded[:, 0], cached


@pytest.mark.parametrize('finish', [True, False])
def test_tensor_beam_finished_and_fallback_share_normalized_definition(finish):
    # Run the actual search with deterministic network outputs. This checks the
    # EOS count at extraction as well as the sequence-limit fallback contract.
    logits = torch.tensor([-20., -20., math.log(.8) if finish else -20., math.log(.2),
                           -20. if finish else math.log(.8)])
    network = SimpleNamespace(
        backbone=lambda img: torch.zeros(img.shape[0], 320, 1, 4),
        encoders=lambda memory, mask: memory,
        decoders=SyntheticDecoders(),
        embd=lambda ids: torch.zeros(*ids.shape, 320),
        pred1=lambda decoded: decoded,
        pred=lambda decoded: logits.expand(decoded.shape[0], -1),
        color_pred1=lambda decoded: decoded,
        color_pred_fg=lambda features: features[..., :3],
        color_pred_bg=lambda features: features[..., :3],
        color_pred_fg_ind=lambda features: features[..., :2],
        color_pred_bg_ind=lambda features: features[..., :2],
    )
    ret = OCR.infer_beam_batch_tensor(network, torch.zeros(1, 3, 48, 8), [8],
                                    beams_k=2, max_finished_hypos=1, max_seq_length=3)[0]
    assert ret[-1] == pytest.approx(.8)
    # The legacy fallback score was already normalized and stays byte-compatible.
    assert ret[1] == pytest.approx(.8 if not finish else .8)


def test_manga_line_confidence_stays_aligned_across_chunks_and_blocks(module):
    from ballontranslator.engine.manga_quality import MANGA_LINE_MIN_CONFIDENCE
    blocks = [block_with_lines(5, manga=True), block_with_lines(4, manga=True)]
    module.set_param_value('chunk_size', 8)
    # Force a saved non-manga setting that would reject every prediction.
    module.set_param_value('min confidence', 1.)
    module.model.model.infer_beam_batch_tensor.side_effect = [
        [reading(.01, MANGA_LINE_MIN_CONFIDENCE)] * 8,
        [reading(.99, .1)],
    ]
    module.run_ocr(np.full((48, 90, 3), 255, np.uint8), blocks)
    assert blocks[0].manga_line_confidence == [MANGA_LINE_MIN_CONFIDENCE] * 5
    assert blocks[1].manga_line_confidence == [MANGA_LINE_MIN_CONFIDENCE] * 3 + [.1]
    assert blocks[0].get_text() == '甲' * 5
    assert blocks[1].get_text() == '甲' * 3
    assert blocks[0].manga_partial is False
    assert blocks[1].manga_partial is True
