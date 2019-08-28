import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertForMultiClass(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiClass, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
