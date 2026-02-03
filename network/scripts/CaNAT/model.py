import torch
import torch.nn as nn
from math import log

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            Tensor with positional encoding applied
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NonAutoregressiveTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.embed = nn.Embedding(input_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_encoder_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_decoder_layers,
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src: torch.Tensor, return_attention_map: bool = False,
                return_embedd: bool = False, return_interm_enc_dec: bool = False, direct_embeddings= False, return_embeddings = False) -> torch.Tensor:
        """
        Arguments:
            src: Input tensor of shape ``[batch_size, seq_len]``

        Returns:
            Output tensor after passing through the model, or attention maps if return_attmap is True.
        """
        if return_embeddings:
            return self.embed(src)
        
        if  direct_embeddings:
            src = src.permute(1, 0, 2)
        else:
            # Embedding and Positional Encoding
            src = self.embed(src).permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)
        
        src = self.pos_encoder(src)

        # Encoding
        enc_output = self.encoder(src)

        # Decoding
        dec_output = self.decoder(src, enc_output)

        if return_embedd:
            return enc_output, dec_output
        # Final linear projection
        dec_output = self.fc(dec_output.permute(1, 0, 2))  # Shape: (batch_size, seq_len, output_dim)

        if return_attention_map:
            l_attenc = []
            l_attdec = []

            # Retrieve encoder self-attentions
            for layer in self.encoder.layers:
                _, attn_weights = layer.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
                l_attenc.append(attn_weights.detach())

            # Retrieve decoder self-attentions
            for layer in self.decoder.layers:
                _, attn_weights = layer.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
                l_attdec.append(attn_weights.detach())

            return l_attenc, l_attdec

        return dec_output
