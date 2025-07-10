from .model_registry import register_model, get_model_by_name, list_available_models
from .fan_layers import (
    FANLayer, 
    FANLayerGated, 
    FANPhaseOffsetLayerGated
)
from .ts_models import (
    TSForecastModel,
    FANForecaster,
    FANGatedForecaster,
    LSTMForecaster,
    TransformerForecaster,
    PhaseOffsetForecaster,
    InterferenceFANForecaster,
    AdaptiveInterferenceFANForecaster,
    HarmonicInterferenceFANForecaster
)

try:
    from ..transformers.fan_transformer import FANTransformerForecaster
    from ..transformers.phase_offset_transformer import PhaseOffsetTransformerForecaster
    from ..transformers.phase_offset_gated_transformer import PhaseOffsetGatedTransformerForecaster
except ImportError:
    # Fallback for local imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from transformers.fan_transformer import FANTransformerForecaster
        from transformers.phase_offset_transformer import PhaseOffsetTransformerForecaster
        from transformers.phase_offset_gated_transformer import PhaseOffsetGatedTransformerForecaster
    except ImportError:
        # If transformers can't be imported, create dummy classes
        class FANTransformerForecaster:
            pass
        class PhaseOffsetTransformerForecaster:
            pass
        class PhaseOffsetGatedTransformerForecaster:
            pass

__all__ = [
    'register_model',
    'get_model_by_name',
    'list_available_models',
    
    'FANLayer',
    'FANLayerGated',
    'FANPhaseOffsetLayerGated',
    
    'TSForecastModel',
    'FANForecaster',
    'FANGatedForecaster',
    'LSTMForecaster',
    'TransformerForecaster',
    'PhaseOffsetForecaster',
    'InterferenceFANForecaster',
    'AdaptiveInterferenceFANForecaster',
    'HarmonicInterferenceFANForecaster',
    
    'FANTransformerForecaster',
    'PhaseOffsetTransformerForecaster',
    'PhaseOffsetGatedTransformerForecaster'
]