from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3PatchEmbeddings,
    LayoutLMv3TextEmbeddings,
    LayoutLMv3Attention,
    LayoutLMv3SelfAttention,
    LayoutLMv3SelfOutput,
    LayoutLMv3Layer,
    LayoutLMv3Encoder,
    LayoutLMv3Intermediate,
    LayoutLMv3Output,
    LayoutLMv3ClassificationHead,
    LayoutLMv3PreTrainedModel,
    LayoutLMv3Model,
)
from transformers.models.layoutlmv3.configuration_layoutlmv3 import LayoutLMv3Config


from torch import nn


class LayoutLMv3ForMLM(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.init_weights()


print("hi")
print(LayoutLMv3ForMLM(LayoutLMv3Config()))
