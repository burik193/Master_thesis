from pathlib import Path
from string import ascii_letters

import numpy as np
import torch
import tensorflow as tf
import plotly.express as px
import sys
from model_parser import create_model
from importlib import reload
from speechbrain.pretrained import Tacotron2
import speechbrain


def drop_nulls(tensor_input, dim=0):
    """
    Drop columns in the input tensor that have variance less than 0.1.

    Args:
        tensor_input (torch.Tensor): Input tensor.
        dim (int, optional): Dimension along which to compute variances. Defaults to 0.

    Returns:
        torch.Tensor: Tensor with dropped null columns.

    """
    tensor_input = tensor_input.cpu()  # Move the input tensor to the CPU
    variances = torch.var(tensor_input, dim=dim)  # Calculate variances along the specified dimension
    mask = variances.abs() > 0.1  # Create a mask to identify columns with variances greater than 0.1

    # Create an index of zero columns
    index_zeros = torch.arange(tensor_input.size(1))[~mask]
    if index_zeros.shape[0] > 1:
        # Find the index of the largest gap between zero columns
        index = index_zeros[torch.diff(index_zeros).argmax() + 1]

        # Select the non-zero columns using the index
        tensor_output = tensor_input[:, :index]
        return tensor_output.unsqueeze(0)  # Return the tensor with dropped null columns
    else:
        return tensor_input.unsqueeze(0)  # Return the input tensor unchanged if no null columns exist


def get_matrix(keyboard: list = ['q w e r t y u i o p', ' a s d f g h j k l ', '\ z x c v b n m , .']) -> np.matrix:
    """
    Convert a list representation of a keyboard layout into a numpy matrix.

    Args:
        keyboard (list, optional): List representing the keyboard layout. Defaults to standard QWERTY layout.

    Returns:
        np.matrix: Numpy matrix representation of the keyboard layout.

    """
    kb = []  # Initialize an empty list to store the rows of the keyboard layout
    for row in keyboard:
        row = [*map(ord, row)]  # Convert each character in the row to its corresponding ASCII code
        kb.append(row)  # Append the row to the keyboard layout list
    return np.matrix(kb, dtype='int')  # Convert the keyboard layout list to a numpy matrix with integer dtype


def get_nearest(matrix: np.matrix, key: str):
    """
    Get the nearest character to a given key in a matrix representing a keyboard layout.

    Args:
        matrix (np.matrix): Matrix representing the keyboard layout.
        key (str): Key for which the nearest character is to be found.

    Returns:
        str: Nearest character to the given key.

    """
    key = key.lower()  # Convert the key to lowercase
    key_d = {key: ord(key)}  # Create a dictionary with the key and its ASCII code
    row_index, col_index = np.where(matrix == key_d[key])  # Find the row and column indices of the key in the matrix

    try:
        row_index, col_index = row_index[0], col_index[0]  # Get the first occurrence of the row and column indices
    except IndexError as e:
        print(f"Row: {row_index}, Col: {col_index}, Key_d: {key}")  # Print debug information
        raise e

    sigma = (0.5 - 1) / 1.96  # Define the standard deviation for random perturbation
    rand_y = round(np.random.normal(0, abs(sigma)))  # Generate a random perturbation value for the row index
    rand_y = -1 if rand_y < -2 else 1 if rand_y > 2 else rand_y  # Apply bounds to the random perturbation value

    rand_x = np.random.rand()  # Generate a random value for the column index perturbation
    rand_x = -2 + abs(rand_y) if rand_x < 0.5 else 2 - abs(rand_y)  # Apply bounds to the random perturbation value

    if not matrix.shape[0] - 1 >= (row_index + rand_y) >= 0:
        rand_y = -1 * rand_y  # Reverse the random perturbation if it causes the row index to go out of bounds

    if not matrix.shape[1] - 1 >= (col_index + rand_x) >= 0:
        rand_x = -1 * rand_x  # Reverse the random perturbation if it causes the column index to go out of bounds

    row_index, col_index = row_index + rand_y, col_index + rand_x  # Apply the random perturbation to the indices

    nearest_values = matrix[row_index, col_index]  # Get the value(s) at the perturbed indices
    vec = np.vectorize(chr)  # Vectorize the chr function to convert ASCII codes to characters
    return vec(nearest_values).item()  # Convert the nearest values to characters and return the result as a string


def common_mistakes(name: str, permutations=True, missclick=True, missing_letter=True, space=True, additional_letter=True) -> list:
    """
    Generate common mistakes or variations of a given name.

    Args:
        name (str): Name for which variations are generated.
        permutations (bool, optional): Whether to generate permutations of adjacent letters. Defaults to True.
        missclick (bool, optional): Whether to generate missclicked versions. Defaults to True.
        missing_letter (bool, optional): Whether to generate versions with a missing letter. Defaults to True.
        space (bool, optional): Whether to generate versions with an added space. Defaults to True.
        additional_letter (bool, optional): Whether to generate versions with an additional letter. Defaults to True.

    Returns:
        tuple: Tuple of generated name variations.

    """
    name = name.replace(' ', '')  # Remove spaces from the name

    missclicked = []
    permutated = []
    missing_l = []
    add_space = []
    additional_l = []

    matrix = get_matrix()  # Get the keyboard matrix
    len_name = len(name)

    for i in range(len_name):
        letter = name[i]

        if missclick and name[i] in ascii_letters:  # Generate missclicked versions
            missclicked.append(name[:i] + get_nearest(matrix, letter) + name[i + 1:])

        if permutations and i + 1 < len_name:  # Generate permutations of adjacent letters
            permutated.append(name[:i] + name[i + 1] + name[i] + name[i + 2:])

        if missing_letter and i + 1 < len_name:  # Generate versions with a missing letter
            missing_l.append(name[:i] + '' + name[i + 1:])

        if space and i + 1 < len_name:  # Generate versions with an added space
            add_space.append(name[:i] + ' ' + name[i:])

        if additional_letter and i + 1 < len_name:  # Generate versions with an additional letter
            additional_l.append(name[:i] + np.random.choice(list(ascii_letters), 1)[0] + name[i:])

    return permutated + missclicked + missing_l + add_space + additional_l  # Return the generated variations as a tuple


def get_spectrogram(waveform):
    """
    Convert the waveform to a spectrogram via a STFT.
    """
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def plot_spectrogram(spectrogram):
    """
    Plot a spectrogram using Plotly.

    Args:
        spectrogram (np.ndarray): Spectrogram data.

    """
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3  # Ensure the spectrogram has the expected shape
        spectrogram = np.squeeze(spectrogram, axis=-1)  # Remove the extra dimension if present

    log_spec = np.log(spectrogram.T + np.finfo(float).eps)  # Compute the logarithm of the spectrogram with epsilon for numerical stability

    fig = px.imshow(log_spec, origin='lower', labels={'x': 'Timeframes', 'y': 'Frequency (kHZ)'})  # Create the spectrogram plot using Plotly
    fig.show()  # Show the spectrogram plot


def import_model(root: str, model_num: int, state: str = 'loss') -> torch.nn.Module:
    """ Rewrite and create a module"""
    root = Path(root).parents[0]
    root = root.joinpath(f'models/architecture_{model_num}')
    with open(root.joinpath('architecture.txt'), 'r', encoding='utf-8') as a:
        architecture = a.read()

    path_replaced = False
    for i, p in enumerate(sys.path):
        if 'architecture' in p:
            sys.path[i] = str(root)
            path_replaced = True
            break
    if not path_replaced:
        sys.path.append(str(root))

    try:
        import forward_pass
        f = reload(forward_pass)
        model = create_model(architecture, forward_method=f.forward)
        print(f'Forward method imported from: {sys.path[-1]}')
    except ImportError:
        model = create_model(architecture)

    # load the state dict
    if state == 'loss':
        state_dict = torch.load(root.joinpath('best_loss_model.pth'))
        print(f'State dict imported for best loss model')
    elif state == 'acc':
        state_dict = torch.load(root.joinpath('best_acc_model.pth'))
        print(f'State dict imported for best acc model')
    else:
        raise Exception(f'Wrong state type specified: expected \"loss\" or \"acc\", got {state}.')

    model.load_state_dict(state_dict)

    return model


# We  override the original Tactoron2 class by modifying the encode_batch() method.
# The original method requires descending sorted data to be passed. We solve this issue here.
# This way we can also employ some interesting ideas of text encoding without worrying that the length of the name changes,
# and it would lead to a crash during generaion phase.

class Tacotron2_modifyed(Tacotron2):
    def encode_batch(self, texts):
        """Computes mel-spectrogram for a list of texts

        Texts must be sorted in decreasing order on their lengths

        Arguments
        ---------
        text: List[str]
            texts to be encoded into spectrogram

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """
        with torch.no_grad():
            text_to_seq = [self.text_to_seq(item) + (item,) for item in texts]
            text_to_seq = sorted(text_to_seq, key=lambda x: x[1], reverse=True)
            inputs = [
                {
                    "text_sequences": torch.tensor(
                        item[0], device=self.device
                    )
                }
                for item in text_to_seq
            ]
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs)

            lens = [item[1] for item in text_to_seq]
            assert lens == sorted(
                lens, reverse=True
            ), "ipnut lengths must be sorted in decreasing order"
            input_lengths = torch.tensor(lens, device=self.device)

            mel_outputs_postnet, mel_lengths, alignments = self.infer(
                inputs.text_sequences.data, input_lengths
            )
            output_texts = [input_text for input_text in text_to_seq]
            batch_len = len(texts)

        return mel_outputs_postnet, output_texts


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False