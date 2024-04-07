import math
import torch
import random
import numpy as np
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr

torch.manual_seed(0)
np.random.seed(0)

def warp(x: torch.Tensor, 
         mu: float = 0.3, 
         std: float = 0.2) -> torch.Tensor:
        
        return torch.normal(mean=mu, std=std, size=x.shape).to(x.device)
        scales = torch.normal(mean=mu, std=std, size=x.shape) + torch.ones(x.shape)
        x = torch.mul(x, scales.to(x.device))
        return x

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the GELU activation function.

    Formula:
        GELU(x) = 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
        equivalent to: GELU(x) = 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
        erf(x) = (2 / sqrt(pi)) * ∫(from 0 to x) e^(-t^2) dt

        Mathematically, these equivalent components produce very similar sigmoid-shaped curves that approximate each other closely,
        especially in the range of values typically encountered in neural network computations.
        As a result, the original GELU formula and the alternative formula produce nearly identical output values for the same input x.

    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class activation(nn.Module):
    def __init__(self, threshold=0.5):
        """
        Layer that zeros out collinear features in a tensor based on the specified threshold.
        Operates on the last two dimensions of a 3D tensor.
        """
        super(activation, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        """
        Forward pass for the layer. Zeroes out collinear features in each sample of the input tensor x.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dimension).

        Returns:
        torch.Tensor: Tensor with collinear features zeroed out for each sample.
        """
        _, sequence_length, _ = x.shape
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_centered = x - x_mean
        cov_matrices = torch.bmm(x_centered.transpose(1, 2), x_centered) / (sequence_length - 1)
        std_devs = torch.sqrt(torch.diagonal(cov_matrices, dim1=1, dim2=2))
        correlation_matrices = cov_matrices / (std_devs[:, :, None] @ std_devs[:, None, :])
        masks = torch.abs(correlation_matrices) > self.threshold
        torch.diagonal(masks, dim1=1, dim2=2).fill_(False)
        indices_to_zero = torch.nonzero(masks, as_tuple=True)
        output = x.clone()
        output[indices_to_zero[0], :, indices_to_zero[2]] = 0.0
        return output
    

class Config(object):
    def __init__(self,
                vocab_size: int,
                hidden_size: int = 768,
                num_hidden_layers: int = 12,
                num_attention_heads: int = 12,
                intermediate_size: int = 3072,
                dropout_prob: float = 0.1,
                max_position_embeddings: int = 512,
                type_vocab_size: int = 2,
                initializer_range: float = 0.02):
        """
        Initialize a configuration object for a model.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int, optional): Size of the hidden layers. Defaults to 768.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 12.
            intermediate_size (int, optional): Size of the intermediate (feed-forward) layers. Defaults to 3072.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.9.
            max_position_embeddings (int, optional): Maximum position embeddings. Defaults to 512.
            type_vocab_size (int, optional): Size of the type vocabulary. Defaults to 2.
            initializer_range (float, optional): Range for weight initialization. Defaults to 0.02.

        Attributes:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Size of the hidden layers.
            num_hidden_layers (int): Number of hidden layers.
            num_attention_heads (int): Number of attention heads.
            intermediate_size (int): Size of the intermediate (feed-forward) layers.
            hidden_dropout_prob (float): Dropout probability for hidden layers.
            attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
            max_position_embeddings (int): Maximum position embeddings.
            type_vocab_size (int): Size of the type vocabulary.
            initializer_range (float): Range for weight initialization.

        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls,
                  dict_object: dict):
        """
        Create a configuration object from a dictionary.

        Args:
            dict_object (dict): A dictionary containing configuration values.

        Returns:
            Config: A Config object initialized from the dictionary.

        """
        config = Config(vocab_size=None)
        for (key, value) in dict_object.items():
            config.__dict__[key] = value
        return config



class LayerNorm(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 variance_epsilon: float = 1e-12):
        """
        Initialize a Layer Normalization module.

        Args:
            hidden_size (int): The size of the hidden layer.
            variance_epsilon (float, optional): A small value added to the denominator for numerical stability.
                Defaults to 1e-12.

        Attributes:
            gamma (nn.Parameter): Learnable scale parameter.
            beta (nn.Parameter): Learnable bias parameter.
            variance_epsilon (float): A small value added to the denominator for numerical stability.

        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Layer Normalization.

        """
        mean = x.mean(dim=-1, keepdim=True)
        std = (x - mean).pow(2).mean(-1, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(std + self.variance_epsilon)
        return self.gamma * x_normalized + self.beta



class MLP(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int):
        """
        Initialize a Multi-Layer Perceptron (MLP) module.

        Args:
            hidden_size (int): The size of the input and output layers.
            intermediate_size (int): The size of the hidden layer.

        Attributes:
            dense_expansion (nn.Linear): Linear transformation for expanding the input.
            dense_contraction (nn.Linear): Linear transformation for contracting the hidden layer.

        """
        super(MLP, self).__init__()
        self.dense_expansion = nn.Linear(hidden_size, intermediate_size)
        self.activation = gelu 
        self.dense_contraction = nn.Linear(intermediate_size, hidden_size)


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size) after passing
                         through the MLP layers.

        """
        x = self.dense_expansion(x)
        x = self.activation(x)
        x = self.dense_contraction(x)

        return x



class Layer(nn.Module):
    def __init__(self,
                 config: 'Config'):
        """
        Initialize a layer module.

        Args:
            config (Config): Configuration object containing model hyperparameters.

        Attributes:
            hidden_size (int): Size of the hidden layers.
            num_attention_heads (int): Number of attention heads.
            attention_head_size (int): Size of each attention head.
            all_head_size (int): Total size of all attention heads.
            query (nn.Linear): Linear layer for query projections.
            key (nn.Linear): Linear layer for key projections.
            value (nn.Linear): Linear layer for value projections.
            dropout (nn.Dropout): Dropout layer for attention probabilities.
            attn_out (nn.Linear): Linear layer for attention output.
            ln1 (LayerNorm): Layer normalization for the first sub-layer.
            mlp (MLP): Multi-Layer Perceptron.
            ln2 (LayerNorm): Layer normalization for the second sub-layer.

        """
        super(Layer, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln1 = LayerNorm(config.hidden_size)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.ln2 = LayerNorm(config.hidden_size)


    def split_heads(self,
                    tensor: torch.Tensor,
                    num_heads: int,
                    attention_head_size: int) -> torch.Tensor:
        """
        Split the input tensor into multiple attention heads.

        Args:
            tensor (torch.Tensor): Input tensor to be split.
            num_heads (int): Number of attention heads.
            attention_head_size (int): Size of each attention head.

        Returns:
            torch.Tensor: Tensor after splitting.

        """
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)


    def merge_heads(self,
                    tensor: torch.Tensor,
                    num_heads: int,
                    attention_head_size: int) -> torch.Tensor:
        """
        Merge attention heads into a single tensor.

        Args:
            tensor (torch.Tensor): Input tensor with separate attention heads.
            num_heads (int): Number of attention heads.
            attention_head_size (int): Size of each attention head.

        Returns:
            torch.Tensor: Tensor after merging.

        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)


    def attn(self,
             q: torch.Tensor,
             k: torch.Tensor,
             v: torch.Tensor,
             attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Tensor after attention.

        Semi-Problem: Does not consider cases where attention mask is not provided
        Solution: Use a condition to chose is to handle mask
        """
        dot_product = torch.matmul(q, k.transpose(-1,-2))
        scaled_dot_product = dot_product / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            mask = attention_mask == 1
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = torch.where(mask, scaled_dot_product, torch.tensor(float('-inf')))
        else:
            attention_scores = scaled_dot_product
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, v)


    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        queries, keys, values = self.query(x), self.key(x), self.value(x)

        queries = self.split_heads(queries, self.num_attention_heads, self.attention_head_size)
        keys = self.split_heads(keys, self.num_attention_heads, self.attention_head_size)
        values = self.split_heads(values, self.num_attention_heads, self.attention_head_size)
        attended_outputs = self.attn(queries, keys, values, attention_mask)
        attended_outputs = self.merge_heads(attended_outputs, self.num_attention_heads, self.attention_head_size)
        attended_outputs = self.attn_out(attended_outputs)
        attended_outputs = self.dropout(attended_outputs)
        normalised_outputs = self.ln1(attended_outputs + x)
        mlp_outputs = self.mlp(normalised_outputs)
        mlp_outputs = self.dropout(mlp_outputs)
        return self.ln2(mlp_outputs + normalised_outputs)



class Bert(nn.Module):
    def __init__(self,
                 config_dict: dict):
        """
        Initialize a BERT model.

        Args:
            config_dict (dict): A dictionary containing model configuration.

        Attributes:
            config (Config): Model configuration object.
            embeddings (nn.ModuleDict): Module dictionary containing token, position, and token_type embeddings.
            ln (LayerNorm): Layer normalization for embeddings.
            dropout (nn.Dropout): Dropout layer.
            layers (nn.ModuleList): List of BERT layers.
            pooler (nn.Sequential): Sequential module for pooler layer.

        """
        super(Bert, self).__init__()
        self.config = Config.from_dict(config_dict)

        self.embeddings = nn.ModuleDict({
            'token': nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0),
            'position': nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
            'token_type': nn.Embedding(self.config.type_vocab_size, self.config.hidden_size),
        })

        self.ln = LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layers = nn.ModuleList([
            Layer(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        self.pooler = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(self.config.hidden_size, self.config.hidden_size)),
            ('activation', nn.Tanh()),
        ]))


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                training: bool = False):
        """
        Forward pass of the BERT model.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            attention_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            token_type_ids (torch.Tensor, optional): Token type IDs tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor and pooled representation.
        """

        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.embeddings['token'](input_ids) + self.embeddings['position'](position_ids)
        x += self.embeddings['token_type'](token_type_ids)
        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        # if training:
        #     for _ in range(len(x)//5):
        #         replace_idx = random.choice(range(x.shape[1]))
        #         x[:, replace_idx, :] = warp(x[:, replace_idx, :])

        return (x, self.pooler(torch.mean(x, dim=1)))


    def load_model(self,
                   path: str):
        """
        Load a pre-trained model from a file.

        Args:
            path (str): Path to the model checkpoint file.

        Returns:
            Bert: The BERT model instance.

        """
        self.load_state_dict(torch.load(path))
        return self