from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class ParallelAdapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_dim: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)
        self.act = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)   
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.up(self.act(self.down(x))))

class _SharedAdapterBank(nn.Module):
    def __init__(self, hidden_size: int, adapter_dim: int, dropout: float, use_mhsa: bool, use_ffn: bool):
        super().__init__()
        self.attn = ParallelAdapter(hidden_size, adapter_dim, dropout) if use_mhsa else None
        self.ffn  = ParallelAdapter(hidden_size, adapter_dim, dropout) if use_ffn  else None

    def attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x) if self.attn is not None else 0.0 * x

    def ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)  if self.ffn  is not None else 0.0 * x

class _W2VLayerWithAdapters(nn.Module):
    def __init__(
        self, base_layer: nn.Module, hidden_size: int, use_mhsa: bool, use_ffn: bool,
        adapter_dim: int, adapter_dropout: float, *, attn_fn=None, ffn_fn=None,
    ):
        super().__init__()
        self.base = base_layer
        self.use_mhsa = use_mhsa
        self.use_ffn = use_ffn
        self._attn_fn = attn_fn
        self._ffn_fn  = ffn_fn
        self.adapter_attn = None
        self.adapter_ffn  = None
        if attn_fn is None and use_mhsa:
            self.adapter_attn = ParallelAdapter(hidden_size, adapter_dim, adapter_dropout)
        if ffn_fn is None and use_ffn:
            self.adapter_ffn  = ParallelAdapter(hidden_size, adapter_dim, adapter_dropout)

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
        if self.use_mhsa:
            if self._attn_fn is not None:
                y = y + self._attn_fn(normed_attn)
            else:
                y = y + self.adapter_attn(normed_attn)

        normed_ffn = self.base.final_layer_norm(y)
        ffn_out = self.base.feed_forward(normed_ffn)
        ffn_out = self.base.dropout(ffn_out)
        z = y + ffn_out
        if self.use_ffn:
            if self._ffn_fn is not None:
                z = z + self._ffn_fn(normed_ffn)
            else:
                z = z + self.adapter_ffn(normed_ffn)

        return z, attn_weights if output_attentions else None

def _adapterize_w2v_stack(
    model: nn.Module, adapter_dim: int, adapter_dropout: float, insertion: Literal["ffn", "mhsa"], shared: bool,
) -> None:
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

    use_mhsa = insertion == "mhsa"
    use_ffn  = insertion == "ffn"
    attn_fn = ffn_fn = None
    if shared:
        bank = _SharedAdapterBank(hidden_size, adapter_dim, adapter_dropout, use_mhsa, use_ffn)
        setattr(model, "_shared_adapters", bank)
        attn_fn = bank.attn_forward if use_mhsa else None
        ffn_fn  = bank.ffn_forward  if use_ffn  else None

    for i in range(len(layers)):
        layers[i] = _W2VLayerWithAdapters(
            base_layer=layers[i], hidden_size=hidden_size, use_mhsa=use_mhsa, use_ffn=use_ffn,
            adapter_dim=adapter_dim, adapter_dropout=adapter_dropout, attn_fn=attn_fn, ffn_fn=ffn_fn,
        )

class _ASTLayerWithAdapters(nn.Module):
    def __init__(
        self, base_layer: nn.Module, hidden_size: int, use_mhsa: bool, use_ffn: bool,
        adapter_dim: int, adapter_dropout: float, *, attn_fn=None, ffn_fn=None,
    ):
        super().__init__()
        self.base = base_layer
        self.use_mhsa = use_mhsa
        self.use_ffn = use_ffn
        self._attn_fn = attn_fn
        self._ffn_fn  = ffn_fn
        self.adapter_attn = None
        self.adapter_ffn  = None
        if attn_fn is None and use_mhsa:
            self.adapter_attn = ParallelAdapter(hidden_size, adapter_dim, adapter_dropout)
        if ffn_fn is None and use_ffn:
            self.adapter_ffn  = ParallelAdapter(hidden_size, adapter_dim, adapter_dropout)

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

        if self.use_mhsa:
            if self._attn_fn is not None:
                y = y + self._attn_fn(normed_attn)
            else:
                y = y + self.adapter_attn(normed_attn)

        normed_ffn = self.base.layernorm_after(y)
        ffn_mid = self.base.intermediate.dense(normed_ffn)
        ffn_mid = self.base.intermediate.intermediate_act_fn(ffn_mid)
        ffn_out = self.base.output.dense(ffn_mid)
        ffn_out = self.base.output.dropout(ffn_out)
        z = y + ffn_out
        if self.use_ffn:
            if self._ffn_fn is not None:
                z = z + self._ffn_fn(normed_ffn)
            else:
                z = z + self.adapter_ffn(normed_ffn)

        return z, attn_weights if output_attentions else None

def _adapterize_ast_stack(
    model: nn.Module, adapter_dim: int, adapter_dropout: float, insertion: Literal["ffn", "mhsa"], shared: bool,
) -> None:
    if not hasattr(model, "audio_spectrogram_transformer"):
        raise AssertionError("AST backbone not found.")
    encoder = model.audio_spectrogram_transformer.encoder
    layers = encoder.layer
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = layers[0].attention.output.dense.out_features

    use_mhsa = insertion == "mhsa"
    use_ffn  = insertion == "ffn"
    attn_fn = ffn_fn = None
    if shared:
        bank = _SharedAdapterBank(hidden_size, adapter_dim, adapter_dropout, use_mhsa, use_ffn)
        setattr(model, "_shared_adapters", bank)
        attn_fn = bank.attn_forward if use_mhsa else None
        ffn_fn  = bank.ffn_forward  if use_ffn  else None

    for i in range(len(layers)):
        layers[i] = _ASTLayerWithAdapters(
            base_layer=layers[i], hidden_size=hidden_size, use_mhsa=use_mhsa, use_ffn=use_ffn,
            adapter_dim=adapter_dim, adapter_dropout=adapter_dropout, attn_fn=attn_fn, ffn_fn=ffn_fn,
        )

class _ViTLayerWithAdapters(nn.Module):
    def __init__(
        self, base_layer: nn.Module, hidden_size: int, use_mhsa: bool, use_ffn: bool,
        adapter_dim: int, adapter_dropout: float, *, attn_fn=None, ffn_fn=None,
    ):
        super().__init__()
        self.base = base_layer
        self.use_mhsa = use_mhsa
        self.use_ffn = use_ffn
        self._attn_fn = attn_fn
        self._ffn_fn  = ffn_fn
        self.adapter_attn = None
        self.adapter_ffn  = None
        if attn_fn is None and use_mhsa:
            self.adapter_attn = ParallelAdapter(hidden_size, adapter_dim, adapter_dropout)
        if ffn_fn is None and use_ffn:
            self.adapter_ffn  = ParallelAdapter(hidden_size, adapter_dim, adapter_dropout)

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
        
        if self.use_mhsa:
            if self._attn_fn is not None:
                y = y + self._attn_fn(normed_attn)
            else:
                y = y + self.adapter_attn(normed_attn)

        normed_ffn = self.base.layernorm_after(y)
        ffn_mid = self.base.intermediate.dense(normed_ffn)
        ffn_mid = self.base.intermediate.intermediate_act_fn(ffn_mid)
        ffn_out = self.base.output.dense(ffn_mid)
        ffn_out = self.base.output.dropout(ffn_out)
        z = y + ffn_out
        
        if self.use_ffn:
            if self._ffn_fn is not None:
                z = z + self._ffn_fn(normed_ffn)
            else:
                z = z + self.adapter_ffn(normed_ffn)

        return z, attn_weights if output_attentions else None

def _adapterize_vit_stack(
    model: nn.Module, adapter_dim: int, adapter_dropout: float, insertion: Literal["ffn", "mhsa"], shared: bool,
) -> None:
    if not hasattr(model, "vit"):
        raise AssertionError("ViT backbone not found.")
    encoder = model.vit.encoder
    layers = encoder.layer
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = layers[0].attention.output.dense.out_features

    use_mhsa = insertion == "mhsa"
    use_ffn  = insertion == "ffn"
    attn_fn = ffn_fn = None
    if shared:
        bank = _SharedAdapterBank(hidden_size, adapter_dim, adapter_dropout, use_mhsa, use_ffn)
        setattr(model, "_shared_adapters", bank)
        attn_fn = bank.attn_forward if use_mhsa else None
        ffn_fn  = bank.ffn_forward  if use_ffn  else None

    for i in range(len(layers)):
        layers[i] = _ViTLayerWithAdapters(
            base_layer=layers[i], hidden_size=hidden_size, use_mhsa=use_mhsa, use_ffn=use_ffn,
            adapter_dim=adapter_dim, adapter_dropout=adapter_dropout, attn_fn=attn_fn, ffn_fn=ffn_fn,
        )

def apply_adapter(
    model: nn.Module, adapter_dim: int = 64, adapter_dropout: float = 0.0,
    insertion: Literal["ffn", "mhsa"] = "ffn", shared: bool = False,
) -> nn.Module:
    cfg: PretrainedConfig = model.config
    mtype = getattr(cfg, "model_type", "").lower()

    if mtype == "audio-spectrogram-transformer":
        _adapterize_ast_stack(model, adapter_dim, adapter_dropout, insertion, shared)
    elif mtype.startswith(("wav2vec2", "hubert", "wavlm")):
        _adapterize_w2v_stack(model, adapter_dim, adapter_dropout, insertion, shared)
    elif mtype == "vit":
        _adapterize_vit_stack(model, adapter_dim, adapter_dropout, insertion, shared)
    else:
        raise NotImplementedError(f"Adapters implemented for model_type in {{'audio-spectrogram-transformer','vit','wav2vec2*','hubert*','wavlm*'}}, got '{mtype}'.")

    for name, p in model.named_parameters():
        nl = name.lower()
        if ("adapter" in nl) or ("_shared_adapters" in nl) or ("classifier" in nl):
            p.requires_grad = True
        else:
            p.requires_grad = False

    return model
