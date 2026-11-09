import numpy as np

from dinov3_mlx.outputs import NUM_PREFIX_TOKENS, split_tokens


def test_split_tokens_drops_register_tokens() -> None:
    height, width, dim = 32, 48, 3
    grid_h, grid_w = height // 16, width // 16
    # Row i holds the value i in every channel.
    tokens = np.repeat(np.arange(NUM_PREFIX_TOKENS + grid_h * grid_w, dtype=np.float32), dim)
    tokens = tokens.reshape(-1, dim)

    frame, patches = split_tokens(tokens, height, width)

    assert NUM_PREFIX_TOKENS == 5
    assert frame.tolist() == [0.0, 0.0, 0.0]
    assert patches.shape == (grid_h, grid_w, dim)
    assert patches[0, 0, 0] == 5.0
    assert patches[0, 1, 0] == 6.0
    assert patches[1, 0, 0] == 5.0 + grid_w
    assert patches.min() >= NUM_PREFIX_TOKENS


def test_split_tokens_keeps_batch_dimension() -> None:
    tokens = np.zeros((2, NUM_PREFIX_TOKENS + 4, 8), dtype=np.float32)
    frame, patches = split_tokens(tokens, 32, 32)
    assert frame.shape == (2, 8)
    assert patches.shape == (2, 2, 2, 8)
