import numpy as np

"""Use utility functions from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/masked_autoregressive.py
"""


def create_input_order(input_size, input_order="left-to-right"):
    """Returns a degree vectors for the input."""
    if input_order == "left-to-right":
        return np.arange(start=1, stop=input_size + 1)
    elif input_order == "right-to-left":
        return np.arange(start=input_size, stop=0, step=-1)
    elif input_order == "random":
        ret = np.arange(start=1, stop=input_size + 1)
        np.random.shuffle(ret)
        return ret


def create_degrees(
    input_size, hidden_units, input_order="left-to-right", hidden_degrees="equal"
):
    input_order = create_input_order(input_size, input_order)
    degrees = [input_order]
    for units in hidden_units:
        if hidden_degrees == "random":
            # samples from: [low, high)
            degrees.append(
                np.random.randint(
                    low=min(np.min(degrees[-1]), input_size - 1),
                    high=input_size,
                    size=units,
                )
            )
        elif hidden_degrees == "equal":
            min_degree = min(np.min(degrees[-1]), input_size - 1)
            degrees.append(
                np.maximum(
                    min_degree,
                    # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
                    # segments, and pick the boundaries between the segments as degrees.
                    np.ceil(
                        np.arange(1, units + 1) * (input_size - 1) / float(units + 1)
                    ).astype(np.int32),
                )
            )
    return degrees


def create_masks(degrees):
    """Returns a list of binary mask matrices enforcing autoregressivity."""
    return [
        # Create input->hidden and hidden->hidden masks.
        inp[:, np.newaxis] <= out
        for inp, out in zip(degrees[:-1], degrees[1:])
    ] + [
        # Create hidden->output mask.
        degrees[-1][:, np.newaxis]
        < degrees[0]
    ]


def check_masks(masks):
    """Check that the connectivity matrix between layers is lower triangular."""
    # (num_input, num_hidden)
    prev = masks[0].t()
    for i in range(1, len(masks)):
        # num_hidden is second axis
        prev = prev @ masks[i].t()
    final = prev.numpy()
    num_input = masks[0].shape[1]
    num_output = masks[-1].shape[0]
    assert final.shape == (num_input, num_output)
    if num_output == num_input:
        assert np.triu(final).all() == 0
    else:
        for submat in np.split(
            final, indices_or_sections=num_output // num_input, axis=1
        ):
            assert np.triu(submat).all() == 0


def build_random_masks(num_input, num_output, num_hidden, num_layers):
    """Build the masks according to Eq 12 and 13 in the MADE paper."""
    # assign input units a number between 1 and D
    rng = np.random.RandomState(0)
    m_list, masks = [], []
    m_list.append(np.arange(1, num_input + 1))
    for i in range(1, num_layers + 1):
        if i == num_layers:
            # assign output layer units a number between 1 and D
            m = np.arange(1, num_input + 1)
            assert (
                num_output % num_input == 0
            ), "num_output must be multiple of num_input"
            m_list.append(np.hstack([m for _ in range(num_output // num_input)]))
        else:
            # assign hidden layer units a number between 1 and D-1
            # i.e. randomly assign maximum number of input nodes to connect to
            m_list.append(rng.randint(1, num_input, size=num_hidden))
        if i == num_layers:
            mask = m_list[i][None, :] > m_list[i - 1][:, None]
        else:
            # input to hidden & hidden to hidden
            mask = m_list[i][None, :] >= m_list[i - 1][:, None]
        # need to transpose for torch linear layer, shape (num_output, num_input)
        masks.append(mask.astype(np.float32).T)
    return masks


def _compute_neighborhood(system_size):
    """Compute (system_size, neighborhood_size) array."""
    num_variables = system_size ** 2
    arange = np.arange(num_variables)
    grid = arange.reshape((system_size, system_size))
    self_and_neighbors = np.zeros((system_size, system_size, 5), dtype=int)
    # four nearest-neighbors
    self_and_neighbors = np.zeros((system_size, system_size, 5), dtype=int)
    self_and_neighbors[..., 0] = grid
    neighbor_index = 1
    for axis in [0, 1]:
        for shift in [-1, 1]:
            self_and_neighbors[..., neighbor_index] = np.roll(
                grid, shift=shift, axis=axis
            )
            neighbor_index += 1
    # reshape to (num_latent, num_neighbors)
    self_and_neighbors = self_and_neighbors.reshape(num_variables, -1)
    return self_and_neighbors


def build_neighborhood_indicator(system_size):
    """Boolean indicator of (num_variables, num_variables) for whether nodes are neighbors."""
    neighborhood = _compute_neighborhood(system_size)
    num_variables = system_size ** 2
    mask = np.zeros((num_variables, num_variables), dtype=bool)
    for i in range(len(mask)):
        mask[i, neighborhood[i]] = True
    return mask


def build_deterministic_mask(num_variables, num_input, num_output, mask_type):
    if mask_type == "input":
        in_degrees = np.arange(num_input) % num_variables
    else:
        in_degrees = np.arange(num_input) % (num_variables - 1)

    if mask_type == "output":
        out_degrees = np.arange(num_output) % num_variables
        mask = np.expand_dims(out_degrees, -1) > np.expand_dims(in_degrees, 0)
    else:
        out_degrees = np.arange(num_output) % (num_variables - 1)
        mask = np.expand_dims(out_degrees, -1) >= np.expand_dims(in_degrees, 0)

    return mask, in_degrees, out_degrees


def build_masks(num_variables, num_input, num_output, num_hidden, mask_fn):
    input_mask, _, _ = mask_fn(num_variables, num_input, num_hidden, "input")
    hidden_mask, _, _ = mask_fn(num_variables, num_hidden, num_hidden, "hidden")
    output_mask, _, _ = mask_fn(num_variables, num_hidden, num_output, "output")
    masks = [input_mask, hidden_mask, output_mask]
    masks = [torch.from_numpy(x.astype(np.float32)) for x in masks]
    return masks


def build_neighborhood_mask(num_variables, num_input, num_output, mask_type):
    system_size = int(np.sqrt(num_variables))
    # return context mask for input, with same assignment of m(k) maximum node degree
    mask, in_degrees, out_degrees = build_deterministic_mask(
        system_size ** 2, num_input, num_output, mask_type
    )
    neighborhood = _compute_neighborhood(system_size)
    neighborhood_mask = np.zeros_like(mask)  # shape len(out_degrees), len(in_degrees)
    for i in range(len(neighborhood_mask)):
        neighborhood_indicator = np.isin(in_degrees, neighborhood[out_degrees[i]])
        neighborhood_mask[i, neighborhood_indicator] = True
    return mask * neighborhood_mask, in_degrees, out_degrees


def checkerboard(shape):
    return (np.indices(shape).sum(0) % 2).astype(np.float32)
