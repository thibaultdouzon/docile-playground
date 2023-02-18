from typing import *

from transformers import (
    LayoutLMv3Config,
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


def get_model(
    model_name: str, proc_kwargs=None, model_kwargs=None
) -> tuple[ProcessorMixin, PreTrainedModel]:
    if proc_kwargs is None:
        proc_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if model_name in model_zoo:
        pretrained_model_name, model_getr = model_zoo[model_name]
        return model_getr(pretrained_model_name, proc_kwargs, model_kwargs)
    else:
        raise NotImplementedError()


def get_layoutlmv3(
    model_name: str, proc_kwargs: dict[str, Any], model_kwargs: dict[str, Any]
):
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_name, **model_kwargs)
    processor = LayoutLMv3Processor.from_pretrained(model_name, **proc_kwargs)
    assert processor is not None
    assert model is not None
    return (processor, model)


model_zoo = {"layoutlmv3-base": ("microsoft/layoutlmv3-base", get_layoutlmv3)}
