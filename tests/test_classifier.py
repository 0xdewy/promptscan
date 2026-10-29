"""Checkpoint identity and complete-input coverage, without downloading weights."""

from types import SimpleNamespace

import torch

from promptscan import classifier as C


class Model:
    config = SimpleNamespace(
        _commit_hash="resolved-commit",
        id2label={0: "SAFE", 1: "INJECTION"},
        promptscan={"threshold": 0.98, "max_length": 512},
    )

    def to(self, device):
        return self

    def eval(self):
        return self


def test_default_model_and_tokenizer_resolve_to_same_commit(monkeypatch):
    calls = []

    def model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return Model()

    def tokenizer(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return object()

    monkeypatch.setattr(C.AutoModelForSequenceClassification, "from_pretrained", model)
    monkeypatch.setattr(C.AutoTokenizer, "from_pretrained", tokenizer)
    clf = C.Classifier.load()
    assert calls[0][2]["revision"] == C.DEFAULT_REVISION
    assert calls[0][2]["use_safetensors"] is True
    assert calls[0][2]["trust_remote_code"] is False
    assert calls[1][2] == {"trust_remote_code": False, "revision": "resolved-commit"}
    assert clf.identity["revision"] == "resolved-commit"
    assert clf.threshold == 0.98


def test_local_identity_changes_with_artifacts_and_ignores_remote_revision(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        C.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda p, **k: calls.append(k) or Model(),
    )
    monkeypatch.setattr(C.AutoTokenizer, "from_pretrained", lambda p, **k: object())
    (tmp_path / "config.json").write_text("one")
    first = C.Classifier.load(str(tmp_path)).identity
    (tmp_path / "config.json").write_text("two")
    second = C.Classifier.load(str(tmp_path)).identity
    assert first["artifact_sha256"] != second["artifact_sha256"]
    assert first["revision"] is None
    assert "revision" not in calls[0]


def test_long_text_is_not_silently_sliced():
    observed = []

    def tokenizer(texts, **kwargs):
        observed.extend(texts)
        return {"overflow_to_sample_mapping": []}

    clf = C.Classifier(Model(), tokenizer, 0.98, 512)
    text = "ordinary " * 25_000 + "tail payload"
    assert clf.score([text]) == [0.0]
    assert observed == [text]
    assert clf.score([]) == []


def test_save_rejects_invalid_metadata_before_writing_files(tmp_path):
    import pytest

    clf = C.Classifier(Model(), object(), 0.98, 512)
    with pytest.raises(ValueError, match="JSON compliant"):
        clf.save(str(tmp_path / "model"), metric=float("nan"))
    assert not (tmp_path / "model").exists()


def test_local_identity_covers_weight_shards_and_tokenizer_assets(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        C.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: Model()
    )
    monkeypatch.setattr(C.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    shard = tmp_path / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"first weights")
    first = C.Classifier.load(str(tmp_path)).identity
    shard.write_bytes(b"changed weights")
    second = C.Classifier.load(str(tmp_path)).identity
    (tmp_path / "special_tokens_map.json").write_text("{}")
    third = C.Classifier.load(str(tmp_path)).identity
    assert first["artifact_sha256"] != second["artifact_sha256"]
    assert second["artifact_sha256"] != third["artifact_sha256"]
    assert shard.name in first["hashed_files"]
    assert "special_tokens_map.json" in third["hashed_files"]


def test_length_sorted_batches_keep_document_ownership_and_maximum():
    from transformers import BatchEncoding

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "overflow_to_sample_mapping": [0, 0, 1],
                "input_ids": [[1, 2, 3], [9], [2, 3]],
                "attention_mask": [[1, 1, 1], [1], [1, 1]],
            }

        def pad(self, values, **kwargs):
            width = max(map(len, values["input_ids"]))
            return BatchEncoding(
                {
                    key: torch.tensor([v + [0] * (width - len(v)) for v in arrays])
                    for key, arrays in values.items()
                }
            )

    class Scoring(Model):
        def __call__(self, input_ids, attention_mask):
            margin = input_ids.max(-1).values.float()
            return SimpleNamespace(
                logits=torch.stack((torch.zeros_like(margin), margin), -1)
            )

    classifier = C.Classifier(Scoring(), Tokenizer(), 0.98, 512)
    result = classifier.score(["long document", "short document"], batch=2)
    expected = torch.tensor([[0.0, 9.0], [0.0, 3.0]]).softmax(-1)[:, 1].tolist()
    assert result == expected
