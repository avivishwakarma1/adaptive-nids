"""Fitted preprocessing artifact (contract sections 3 and 8).

Fit ONLY on training data. The artifact is bundled with the model and carries the exact
ordered input/output feature names; `transform` refuses frames whose columns differ.

Numeric: log1p -> median impute (fitted) -> standard scale, plus a missing-indicator column per
         numeric feature (so "missing" stays visible after imputation).
Categorical: vocabulary fitted on training data (frequency >= min_category_count, capped at
         max_categories_per_column), one-hot encoded; everything else - rare *and* unseen values - goes to
         the always-present `__other__` column.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .contracts import DEFAULT_MAX_BATCH_SIZE, ConfigError, InputValidationError, SchemaVersionError, SUPPORTED_SCHEMA_VERSIONS
from .features import (
    CATEGORICAL_COLUMNS, MODEL_INPUT_COLUMNS, NUMERIC_COLUMNS, build_model_frame, validate_features,
)

PREPROCESSING_VERSION = "1.0.0"
OTHER_CATEGORY = "__other__"


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_category_count: int = Field(default=5, ge=1)
    max_categories_per_column: int = Field(default=50, ge=2)


class VocabularyOneHot(BaseEstimator, TransformerMixin):
    """Fitted-vocabulary one-hot encoder with a GUARANTEED `__other__` column per categorical feature.

    Values that are rare in training (count < min_count / beyond max_categories) and values never seen at
    fit time both land in `__other__`, so unseen categories can never crash inference or silently
    become an all-zero row.
    """

    def __init__(self, min_count: int = 5, max_categories: int = 50):
        self.min_count = min_count
        self.max_categories = max_categories

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = [str(c) for c in df.columns]
        self.vocab_: dict[str, list[str]] = {}
        self.seen_: dict[str, frozenset] = {}
        for c in df.columns:
            vc = df[c].astype(str).value_counts()
            items = sorted(((v, int(n)) for v, n in vc.items()), key=lambda t: (-t[1], t[0]))  # deterministic
            keep = [v for v, n in items if n >= self.min_count and v != OTHER_CATEGORY][: self.max_categories - 1]
            self.vocab_[str(c)] = sorted(keep)
            self.seen_[str(c)] = frozenset(v for v, _ in items)
        self.categories_ = [self.vocab_[c] + [OTHER_CATEGORY] for c in self.columns_]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        blocks = []
        for c, cats in zip(self.columns_, self.categories_):
            index = {v: i for i, v in enumerate(cats)}
            other = index[OTHER_CATEGORY]
            codes = np.fromiter((index.get(v, other) for v in df[c].astype(str)), dtype=np.int64, count=len(df))
            block = np.zeros((len(df), len(cats)), dtype=np.float64)
            block[np.arange(len(df)), codes] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    def unseen_mask(self, X) -> np.ndarray:
        df = pd.DataFrame(X)
        m = np.zeros(len(df), dtype=bool)
        for c in df.columns:
            m |= ~df[c].astype(str).isin(self.seen_[str(c)]).to_numpy()
        return m


def _log1p(x):
    return np.log1p(x)


class PreprocessingArtifact:
    """Fitted, picklable preprocessing pipeline + metadata. Build with `fit_preprocessing`."""

    def __init__(self, *, schema_version: str, config: PreprocessingConfig, transformer: ColumnTransformer,
                 input_columns: list[str], feature_names_out: list[str], fit_n_rows: int,
                 fitted_statistics: dict[str, Any]):
        self.schema_version = schema_version
        self.preprocessing_version = PREPROCESSING_VERSION
        self.config = config
        self.transformer = transformer
        self.input_columns = list(input_columns)
        self.feature_names_out = list(feature_names_out)
        self.fit_n_rows = fit_n_rows
        self.fitted_statistics = fitted_statistics

    # -- helpers ----------------------------------------------------------------
    @property
    def _mapper(self) -> VocabularyOneHot:
        return self.transformer.named_transformers_["cat"]

    def _check_frame(self, frame: pd.DataFrame) -> None:
        if list(frame.columns) != self.input_columns:
            raise InputValidationError("model frame columns do not match the artifact's ordered feature list")

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_frame(frame)
        X = self.transformer.transform(frame)
        X = np.asarray(X, dtype="float64")
        if X.shape[1] != len(self.feature_names_out):
            raise InputValidationError("transformed width does not match the artifact's feature list")
        if not np.all(np.isfinite(X)):
            raise InputValidationError("preprocessing produced non-finite values")
        return X

    def unseen_category_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """True where a categorical value was never seen when the artifact was fitted."""
        self._check_frame(frame)
        return self._mapper.unseen_mask(frame[list(CATEGORICAL_COLUMNS)])

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "preprocessing_version": self.preprocessing_version,
            "input_columns": self.input_columns, "feature_names_out": self.feature_names_out,
            "fit_n_rows": self.fit_n_rows, "config": self.config.model_dump(),
            "category_vocabulary": {k: list(v) for k, v in self._mapper.vocab_.items()},
            "numeric_medians_log1p_space": self.fitted_statistics.get("numeric_medians"),
        }


def fit_preprocessing(train_frame: pd.DataFrame, config: Optional[PreprocessingConfig] = None,
                      schema_version: str = "1.0.0") -> PreprocessingArtifact:
    """Fit preprocessing on the TRAINING frame only (never on validation/test data)."""
    config = config or PreprocessingConfig()
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(f"unsupported feature schema version {schema_version!r}")
    if list(train_frame.columns) != list(MODEL_INPUT_COLUMNS):
        raise ConfigError("training frame columns must equal MODEL_INPUT_COLUMNS in order")
    if len(train_frame) == 0:
        raise ConfigError("cannot fit preprocessing on an empty frame")

    num_pipe = Pipeline([
        ("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cat_enc = VocabularyOneHot(config.min_category_count, config.max_categories_per_column)
    ct = ColumnTransformer(
        [("num", num_pipe, list(NUMERIC_COLUMNS)),
         ("missing", MissingIndicator(features="all"), list(NUMERIC_COLUMNS)),
         ("cat", cat_enc, list(CATEGORICAL_COLUMNS))],
        remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False,
    )
    ct.fit(train_frame)

    enc: VocabularyOneHot = ct.named_transformers_["cat"]
    names = [f"num__{c}" for c in NUMERIC_COLUMNS] + [f"missing__{c}" for c in NUMERIC_COLUMNS]
    for col, cats in zip(CATEGORICAL_COLUMNS, enc.categories_):
        names += [f"cat__{col}={v}" for v in cats]
    medians = ct.named_transformers_["num"].named_steps["impute"].statistics_
    return PreprocessingArtifact(
        schema_version=schema_version, config=config, transformer=ct,
        input_columns=list(MODEL_INPUT_COLUMNS), feature_names_out=names, fit_n_rows=len(train_frame),
        fitted_statistics={"numeric_medians": {c: float(m) for c, m in zip(NUMERIC_COLUMNS, medians)}},
    )


def preprocess(records: Any, artifact: PreprocessingArtifact, schema_version: str = "1.0.0") -> np.ndarray:
    """Contract entry point: validate raw canonical records, build the model frame, transform.

    Returns a float64 matrix whose columns are exactly ``artifact.feature_names_out``.
    Raises `InputValidationError` / `SchemaVersionError`; never returns partial output.
    """
    if schema_version != artifact.schema_version:
        raise SchemaVersionError("requested schema version does not match the preprocessing artifact")
    report = validate_features(records, schema_version, max_batch_size=DEFAULT_MAX_BATCH_SIZE)
    report.raise_if_invalid()
    frame = build_model_frame([r for r in report.records if r is not None])
    return artifact.transform_frame(frame)
