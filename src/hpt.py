from __future__ import annotations
from typing import Literal, Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from src.RBFHistogramPooling import HistogramLayer

def _init_histogram_layer(hist, width_init: float = 1.0):
    nn.init.zeros_(hist.bin_centers_conv.bias)
    with torch.no_grad():
        hist.bin_widths_conv.weight.fill_(width_init)

class HistogramAdapterParallel(nn.Module):
    def __init__(self, hidden_size: int, num_bins: int, normalize_count: bool = True, normalize_bins: bool = True, init_rand: bool = False):
        super().__init__()
        assert hidden_size % num_bins == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_bins ({num_bins}) "
            "because no projection back to hidden_size is allowed."
        )
        out_size = hidden_size // num_bins
        self.hidden_size = hidden_size
        self.num_bins = num_bins

        self.hist = HistogramLayer(
            in_channels=hidden_size, kernel_size=1, dim=1, num_bins=num_bins,
            stride=1, padding=0, normalize_count=normalize_count,
            normalize_bins=normalize_bins, count_include_pad=False,
            ceil_mode=False, output_size=out_size,
        )
        if not init_rand:
            _init_histogram_layer(self.hist, width_init=1.0)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_preln: torch.Tensor) -> torch.Tensor:
        B, T, H = x_preln.shape
        h = self.hist(x_preln.transpose(1, 2))  
        h_flat = h.reshape(B, -1)               
        h_broadcast = h_flat.unsqueeze(1).expand(B, T, H)  
        return h_broadcast * self.gate

class _W2VLayerWithHPT(nn.Module):
    def __init__(
        self, base_layer: nn.Module, hidden_size: int,
        hpt_mhsa: Optional[HistogramAdapterParallel],
        hpt_ffn: Optional[HistogramAdapterParallel],
    ):
        super().__init__()
        self.base = base_layer
        self.hpt_mhsa = hpt_mhsa
        self.hpt_ffn = hpt_ffn

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False, **kwargs,
    ):
        normed_attn = self.base.layer_norm(hidden_states)
        attn_outputs = self.base.attention(
            hidden_states=normed_attn,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_out = attn_outputs[0]
        attn_weights = attn_outputs[1] if (output_attentions and len(attn_outputs) > 1) else None

        attn_out = self.base.dropout(attn_out)
        y = hidden_states + attn_out
        if self.hpt_mhsa is not None:
            y = y + self.hpt_mhsa(normed_attn)

        normed_ffn = self.base.final_layer_norm(y)
        ffn_out = self.base.feed_forward(normed_ffn)
        ffn_out = self.base.dropout(ffn_out)
        z = y + ffn_out
        if self.hpt_ffn is not None:
            z = z + self.hpt_ffn(normed_ffn)

        return z, attn_weights if output_attentions else None

def _hptize_w2v_stack(
    model: nn.Module, num_bins: int, location: Literal["mhsa", "ffn", "mhsa_ffn"],
    shared: bool, init_rand: bool, external_hpt_modules: Optional[List[HistogramAdapterParallel]] = None,
) -> List[HistogramAdapterParallel]:
    encoder = None
    for attr in ("wav2vec2", "hubert", "wavlm"):
        if hasattr(model, attr):
            enc_parent = getattr(model, attr)
            if hasattr(enc_parent, "encoder"):
                encoder = enc_parent.encoder
                break
    if encoder is None or not hasattr(encoder, "layers"):
        raise AssertionError("Wav2Vec2/HubERT/WavLM encoder not found.")

    layers = encoder.layers
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = layers[0].attention.out_proj.out_features

    use_mhsa = location in ("mhsa", "mhsa_ffn")
    use_ffn  = location in ("ffn", "mhsa_ffn")
    created_modules = []
    ext_idx = 0
    hpt_mhsa_shared = hpt_ffn_shared = None

    if shared:
        if external_hpt_modules is not None:
            if use_mhsa:
                hpt_mhsa_shared = external_hpt_modules[ext_idx]
                ext_idx += 1
            if use_ffn:
                hpt_ffn_shared = external_hpt_modules[ext_idx]
                ext_idx += 1
        else:
            hpt_mhsa_shared = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_mhsa else None
            hpt_ffn_shared  = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_ffn else None
            if hpt_mhsa_shared: created_modules.append(hpt_mhsa_shared)
            if hpt_ffn_shared: created_modules.append(hpt_ffn_shared)

    for i in range(len(layers)):
        if shared:
            hpt_mhsa = hpt_mhsa_shared
            hpt_ffn = hpt_ffn_shared
        else:
            if external_hpt_modules is not None:
                hpt_mhsa = external_hpt_modules[ext_idx] if use_mhsa else None
                if use_mhsa: ext_idx += 1
                hpt_ffn = external_hpt_modules[ext_idx] if use_ffn else None
                if use_ffn: ext_idx += 1
            else:
                hpt_mhsa = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_mhsa else None
                hpt_ffn  = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_ffn else None
                if hpt_mhsa: created_modules.append(hpt_mhsa)
                if hpt_ffn: created_modules.append(hpt_ffn)

        layers[i] = _W2VLayerWithHPT(
            base_layer=layers[i], hidden_size=hidden_size,
            hpt_mhsa=hpt_mhsa, hpt_ffn=hpt_ffn,
        )
        
    return external_hpt_modules if external_hpt_modules is not None else created_modules

class _ASTLayerWithHPT(nn.Module):
    def __init__(
        self, base_layer: nn.Module, hidden_size: int,
        hpt_mhsa: Optional[HistogramAdapterParallel],
        hpt_ffn: Optional[HistogramAdapterParallel],
    ):
        super().__init__()
        self.base = base_layer
        self.hpt_mhsa = hpt_mhsa
        self.hpt_ffn = hpt_ffn

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False, output_attentions: bool = False, **kwargs,
    ):
        normed_attn = self.base.layernorm_before(hidden_states)
        attn_outputs = self.base.attention(
            normed_attn, head_mask=head_mask, output_attentions=output_attentions
        )
        attn_out = attn_outputs[0]
        attn_weights = attn_outputs[1] if (output_attentions and len(attn_outputs) > 1) else None
        attn_out = self.base.attention.output.dropout(attn_out)
        y = hidden_states + attn_out

        if self.hpt_mhsa is not None:
            y = y + self.hpt_mhsa(normed_attn)

        normed_ffn = self.base.layernorm_after(y)
        ffn_mid = self.base.intermediate.dense(normed_ffn)
        ffn_mid = self.base.intermediate.intermediate_act_fn(ffn_mid)
        ffn_out = self.base.output.dense(ffn_mid)
        ffn_out = self.base.output.dropout(ffn_out)
        z = y + ffn_out
        if self.hpt_ffn is not None:
            z = z + self.hpt_ffn(normed_ffn)

        return z, attn_weights if output_attentions else None

def _hptize_ast_stack(
    model: nn.Module, num_bins: int, location: Literal["mhsa", "ffn", "mhsa_ffn"],
    shared: bool, init_rand: bool, external_hpt_modules: Optional[List[HistogramAdapterParallel]] = None,
) -> List[HistogramAdapterParallel]:
    if not hasattr(model, "audio_spectrogram_transformer"):
        raise AssertionError("AST backbone not found.")
    encoder = model.audio_spectrogram_transformer.encoder
    layers = encoder.layer
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = layers[0].attention.output.dense.out_features

    use_mhsa = location in ("mhsa", "mhsa_ffn")
    use_ffn  = location in ("ffn", "mhsa_ffn")
    created_modules = []
    ext_idx = 0
    hpt_mhsa_shared = hpt_ffn_shared = None

    if shared:
        if external_hpt_modules is not None:
            if use_mhsa:
                hpt_mhsa_shared = external_hpt_modules[ext_idx]
                ext_idx += 1
            if use_ffn:
                hpt_ffn_shared = external_hpt_modules[ext_idx]
                ext_idx += 1
        else:
            hpt_mhsa_shared = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_mhsa else None
            hpt_ffn_shared  = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_ffn else None
            if hpt_mhsa_shared: created_modules.append(hpt_mhsa_shared)
            if hpt_ffn_shared: created_modules.append(hpt_ffn_shared)

    for i in range(len(layers)):
        if shared:
            hpt_mhsa = hpt_mhsa_shared
            hpt_ffn = hpt_ffn_shared
        else:
            if external_hpt_modules is not None:
                hpt_mhsa = external_hpt_modules[ext_idx] if use_mhsa else None
                if use_mhsa: ext_idx += 1
                hpt_ffn = external_hpt_modules[ext_idx] if use_ffn else None
                if use_ffn: ext_idx += 1
            else:
                hpt_mhsa = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_mhsa else None
                hpt_ffn  = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_ffn else None
                if hpt_mhsa: created_modules.append(hpt_mhsa)
                if hpt_ffn: created_modules.append(hpt_ffn)
                
        layers[i] = _ASTLayerWithHPT(
            base_layer=layers[i], hidden_size=hidden_size,
            hpt_mhsa=hpt_mhsa, hpt_ffn=hpt_ffn,
        )
        
    return external_hpt_modules if external_hpt_modules is not None else created_modules

class _ViTLayerWithHPT(nn.Module):
    def __init__(
        self, base_layer: nn.Module, hidden_size: int,
        hpt_mhsa: Optional[HistogramAdapterParallel],
        hpt_ffn: Optional[HistogramAdapterParallel],
    ):
        super().__init__()
        self.base = base_layer
        self.hpt_mhsa = hpt_mhsa
        self.hpt_ffn = hpt_ffn

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False, **kwargs,
    ):
        normed_attn = self.base.layernorm_before(hidden_states)
        attn_outputs = self.base.attention(
            normed_attn, head_mask=head_mask, output_attentions=output_attentions
        )
        attn_out = attn_outputs[0]
        attn_weights = attn_outputs[1] if (output_attentions and len(attn_outputs) > 1) else None
        y = hidden_states + attn_out

        if self.hpt_mhsa is not None:
            y = y + self.hpt_mhsa(normed_attn)

        normed_ffn = self.base.layernorm_after(y)
        ffn_mid = self.base.intermediate.dense(normed_ffn)
        ffn_mid = self.base.intermediate.intermediate_act_fn(ffn_mid)
        ffn_out = self.base.output.dense(ffn_mid)
        ffn_out = self.base.output.dropout(ffn_out)
        z = y + ffn_out
        if self.hpt_ffn is not None:
            z = z + self.hpt_ffn(normed_ffn)

        return z, attn_weights if output_attentions else None

def _hptize_vit_stack(
    model: nn.Module, num_bins: int, location: Literal["mhsa", "ffn", "mhsa_ffn"],
    shared: bool, init_rand: bool, external_hpt_modules: Optional[List[HistogramAdapterParallel]] = None,
) -> List[HistogramAdapterParallel]:
    if not hasattr(model, "vit"):
        raise AssertionError("ViT backbone not found.")
    encoder = model.vit.encoder
    layers = encoder.layer
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = layers[0].attention.output.dense.out_features

    use_mhsa = location in ("mhsa", "mhsa_ffn")
    use_ffn  = location in ("ffn", "mhsa_ffn")
    created_modules = []
    ext_idx = 0
    hpt_mhsa_shared = hpt_ffn_shared = None

    if shared:
        if external_hpt_modules is not None:
            if use_mhsa:
                hpt_mhsa_shared = external_hpt_modules[ext_idx]
                ext_idx += 1
            if use_ffn:
                hpt_ffn_shared = external_hpt_modules[ext_idx]
                ext_idx += 1
        else:
            hpt_mhsa_shared = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_mhsa else None
            hpt_ffn_shared  = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_ffn else None
            if hpt_mhsa_shared: created_modules.append(hpt_mhsa_shared)
            if hpt_ffn_shared: created_modules.append(hpt_ffn_shared)

    for i in range(len(layers)):
        if shared:
            hpt_mhsa = hpt_mhsa_shared
            hpt_ffn = hpt_ffn_shared
        else:
            if external_hpt_modules is not None:
                hpt_mhsa = external_hpt_modules[ext_idx] if use_mhsa else None
                if use_mhsa: ext_idx += 1
                hpt_ffn = external_hpt_modules[ext_idx] if use_ffn else None
                if use_ffn: ext_idx += 1
            else:
                hpt_mhsa = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_mhsa else None
                hpt_ffn  = HistogramAdapterParallel(hidden_size, num_bins, init_rand=init_rand) if use_ffn else None
                if hpt_mhsa: created_modules.append(hpt_mhsa)
                if hpt_ffn: created_modules.append(hpt_ffn)
                
        layers[i] = _ViTLayerWithHPT(
            base_layer=layers[i], hidden_size=hidden_size,
            hpt_mhsa=hpt_mhsa, hpt_ffn=hpt_ffn,
        )

    return external_hpt_modules if external_hpt_modules is not None else created_modules

def apply_hpt(
    model: nn.Module, num_bins: int = 4, location: Literal["mhsa", "ffn", "mhsa_ffn"] = "mhsa",
    shared: bool = False, init_rand: bool = False, freeze_others: bool = True,
    external_hpt_modules: Optional[List[HistogramAdapterParallel]] = None,
) -> Tuple[nn.Module, List[HistogramAdapterParallel]]:
    cfg: PretrainedConfig = model.config
    mtype = getattr(cfg, "model_type", "").lower()

    if mtype == "audio-spectrogram-transformer":
        hpt_modules = _hptize_ast_stack(model, num_bins, location, shared, init_rand, external_hpt_modules)
    elif mtype.startswith(("wav2vec2", "hubert", "wavlm")):
        hpt_modules = _hptize_w2v_stack(model, num_bins, location, shared, init_rand, external_hpt_modules)
    elif mtype == "vit":
        hpt_modules = _hptize_vit_stack(model, num_bins, location, shared, init_rand, external_hpt_modules)
    else:
        raise NotImplementedError(
            f"HPT implemented for model_type in "
            f"{{'audio-spectrogram-transformer','vit','wav2vec2*','hubert*','wavlm*'}}, got '{mtype}'."
        )

    if freeze_others:
        for name, p in model.named_parameters():
            name_l = name.lower()
            if ("hpt" in name_l) or ("hist" in name_l) or ("gate" in name_l) or ("classifier" in name_l):
                p.requires_grad = True
            else:
                p.requires_grad = False

    return model, hpt_modules
