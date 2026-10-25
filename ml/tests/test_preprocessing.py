import numpy as np
import pandas as pd
import pytest

from nids_ml import InputValidationError, SchemaVersionError, preprocess
from nids_ml.features import MODEL_INPUT_COLUMNS, build_model_frame, validate_features
from nids_ml.preprocessing import OTHER_CATEGORY, PreprocessingConfig, fit_preprocessing
from test_features import GOOD, rec


def _frame(rows):
    rep = validate_features(rows, max_batch_size=None); rep.raise_if_invalid()
    return build_model_frame(rep.records)


def _train_rows(n=60):
    return [rec(duration=0.1 * (i + 1), flow_bytes=100.0 * (i + 1), connection_state="sf" if i % 2 else "s0",
                destination_port=80 if i % 3 else 443) for i in range(n)]


def test_fit_is_deterministic_and_transform_repeatable():
    f = _frame(_train_rows())
    a1, a2 = fit_preprocessing(f), fit_preprocessing(f)
    assert a1.feature_names_out == a2.feature_names_out
    np.testing.assert_array_equal(a1.transform_frame(f), a2.transform_frame(f))
    np.testing.assert_array_equal(a1.transform_frame(f), a1.transform_frame(f))


def test_output_names_order_and_width_match_artifact():
    f = _frame(_train_rows()); art = fit_preprocessing(f)
    X = art.transform_frame(f)
    assert X.shape == (len(f), len(art.feature_names_out)) and np.isfinite(X).all()
    assert art.input_columns == list(MODEL_INPUT_COLUMNS)
    assert art.feature_names_out[:5] == [f"num__{c}" for c in MODEL_INPUT_COLUMNS[:5]]


def test_column_order_mismatch_is_refused():
    f = _frame(_train_rows()); art = fit_preprocessing(f)
    with pytest.raises(InputValidationError):
        art.transform_frame(f[list(reversed(f.columns))])
    with pytest.raises(InputValidationError):
        art.transform_frame(f.drop(columns=["direction"]))


def test_imputation_uses_training_median_not_inference_data():
    train = _frame(_train_rows())
    art = fit_preprocessing(train)
    med = art.fitted_statistics["numeric_medians"]["flow_bytes"]
    miss = _frame([rec(flow_bytes=None)])
    ref = _frame([rec(flow_bytes=float(np.expm1(med)))])
    Xm, Xr = art.transform_frame(miss), art.transform_frame(ref)
    i_val = art.feature_names_out.index("num__flow_bytes"); i_ind = art.feature_names_out.index("missing__flow_bytes")
    assert Xm[0, i_val] == pytest.approx(Xr[0, i_val])         # imputed with the TRAIN median
    assert Xm[0, i_ind] == 1.0 and Xr[0, i_ind] == 0.0         # missingness stays visible


def test_fit_uses_only_the_frame_it_is_given_no_leakage():
    train = _frame(_train_rows())
    other = _frame([rec(duration=9999.0, protocol="icmp", destination_port=31337)] * 30)
    a = fit_preprocessing(train)
    _ = a.transform_frame(other)                      # transforming other data...
    b = fit_preprocessing(train)
    assert a.describe()["numeric_medians_log1p_space"] == b.describe()["numeric_medians_log1p_space"]
    assert a.describe()["category_vocabulary"] == b.describe()["category_vocabulary"]
    assert "icmp" not in a.describe()["category_vocabulary"]["protocol"]
    assert "31337" not in a.describe()["category_vocabulary"]["destination_port"]


def test_unseen_and_rare_categories_map_to_other_without_crashing():
    art = fit_preprocessing(_frame(_train_rows()), PreprocessingConfig(min_category_count=5))
    f = _frame([rec(protocol="icmp", connection_state="never-seen", destination_port=65000, direction="internal")])
    X = art.transform_frame(f)
    assert np.isfinite(X).all()
    assert art.unseen_category_mask(f)[0]
    names = art.feature_names_out
    assert X[0, names.index(f"cat__protocol={OTHER_CATEGORY}")] == 1.0
    assert X[0, names.index(f"cat__connection_state={OTHER_CATEGORY}")] == 1.0
    seen = _frame([rec(connection_state="sf", destination_port=80)])
    assert not art.unseen_category_mask(seen)[0]


def test_preprocess_entry_point_validates_and_checks_versions(outcome):
    art = outcome.bundle.preprocessing
    X = preprocess([GOOD], art)
    assert X.shape == (1, len(art.feature_names_out))
    with pytest.raises(InputValidationError):
        preprocess([], art)
    with pytest.raises(InputValidationError):
        preprocess([rec(duration=-1)], art)
    with pytest.raises(SchemaVersionError):
        preprocess([GOOD], art, schema_version="9.9.9")


def test_all_missing_numeric_column_does_not_break_fit():
    rows = [rec(flow_packets=None) for _ in range(30)]
    art = fit_preprocessing(_frame(rows))
    assert np.isfinite(art.transform_frame(_frame(rows[:2]))).all()
