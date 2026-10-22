"""Wrapper for an off-the-shelf Hugging Face injection classifier (optional, not part of the ensemble)."""

from typing import Any, Dict, Optional

import torch

from .base_model import BaseModel, BaseProcessor, finish_prediction

DEFAULT_MODEL = "deepset/deberta-v3-base-injection"


class PretrainedInjectionModel(BaseModel):
    """Any HF sequence-classification model whose label 1 (or label "INJECTION") means injection."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        labels = {v.upper(): k for k, v in self.model.config.id2label.items()}
        self.injection_index = labels.get("INJECTION", 1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.model(**inputs).logits
        if self.injection_index != 1:
            logits = logits[:, [1 - self.injection_index, self.injection_index]]
        return logits

    def predict(
        self,
        text: str,
        processor: BaseProcessor,
        confidence_threshold: float = 0.5,
        calibration_temperature: float = 1.0,
    ) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            inputs = {k: v.to(self.get_device()) for k, v in processor.encode(text).items()}
            for key in list(inputs):
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key].unsqueeze(0)
            outputs = self(inputs)
        return finish_prediction(outputs, "pretrained", confidence_threshold, calibration_temperature)

    @classmethod
    def load(cls, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        from ..processors.subword_processor import SubwordProcessor

        model = cls(checkpoint_path or DEFAULT_MODEL)
        model.to(device).eval()
        return model, SubwordProcessor(model_name=model.model_name, max_length=512)

    def save(self, checkpoint_path: str, processor=None, **metadata):
        self.model.save_pretrained(checkpoint_path)
