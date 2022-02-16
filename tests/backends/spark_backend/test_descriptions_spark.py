import datetime

import numpy as np
import pandas as pd
import pytest

from pandas_profiling.config import SparkSettings
from pandas_profiling.model.describe import describe
from pandas_profiling.model.summary import *
from pandas_profiling.model.typeset import ProfilingTypeSet

check_is_NaN = "pandas_profiling.check_is_NaN"


@pytest.fixture
def describe_data():
    data = {
        "id": [chr(97 + c) for c in range(1, 9)] + ["d"],
        "x": [50, 50, -10, 0, 0, 5, 15, -3, np.nan],
        "y": [
            0.000001,
            654.152,
            np.nan,
            15.984512,
            3122,
            -3.1415926535,
            111,
            15.9,
            13.5,
        ],
        "cat": [
            "a",
            "long text value",
            "Élysée",
            "",
            None,
            "some <b> B.s </div> </div> HTML stuff",
            "c",
            "c",
            "c",
        ],
        "s1": np.ones(9),
        "s2": ["some constant text $ % value {obj} " for _ in range(1, 10)],
        "somedate": [
            datetime.date(2011, 7, 4),
            datetime.datetime(2022, 1, 1, 13, 57),
            datetime.datetime(1990, 12, 9),
            np.nan,
            datetime.datetime(1990, 12, 9),
            datetime.datetime(1970, 12, 9),
            datetime.datetime(1972, 1, 2),
            datetime.datetime(1970, 12, 9),
            datetime.datetime(1970, 12, 9),
        ],
        "bool_tf": [True, True, False, True, False, True, True, False, True],
        "bool_tf_with_nan": [
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            np.nan,
        ],
        "bool_01": [1, 1, 0, 1, 1, 0, 0, 0, 1],
        "bool_01_with_nan": [1, 0, 1, 0, 0, 1, 1, 0, np.nan],
        "list": [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ],
        "mixed": [1, 2, "a", 4, 5, 6, 7, 8, 9],
        "dict": [{"hello": "there", "General": "Kenobi"}],
    }
    return data


@pytest.fixture
def expected_results():
    return {
        "id": {
            "count": 9,
            "cv": check_is_NaN,
            "n_distinct": 8,
            "histogram": check_is_NaN,
            "iqr": check_is_NaN,
            "is_unique": False,
            "kurtosis": check_is_NaN,
            "mad": check_is_NaN,
            "max": check_is_NaN,
            "mean": check_is_NaN,
            "min": check_is_NaN,
            "mini_histogram": check_is_NaN,
            "n_missing": 0,
            "p_missing": 0.0,
            "p_distinct": 0.88888888,
            "p_zeros": check_is_NaN,
            "range": check_is_NaN,
            "skewness": check_is_NaN,
            "std": check_is_NaN,
            "sum": check_is_NaN,
            "variance": check_is_NaN,
        },
        "x": {
            "n": 9,
            "count": 9,
            "p_missing": 0.0,
            "n_distinct": 7,
            "n_unique": 5,
            "p_distinct": 0.7777777777777778,
            "is_unique": False,
            "p_unique": 0.5555555555555556,
            "n_infinite": 0,
            "p_infinite": 0.0,
            "n_zeros": 2,
            "p_zeros": 0.2222222222222222,
            "n_negative": 2,
            "p_negative": 0.2222222222222222,
            "quantiles": {
                "5%": -10.0,
                "25%": -3.0,
                "50%": 0.0,
                "75%": 15.0,
                "95%": 50.0,
            },
            "mad": 5.0,
            "min": -10.0,
            "max": check_is_NaN,
            "mean": check_is_NaN,
            "std": check_is_NaN,
            "variance": check_is_NaN,
            "kurtosis": check_is_NaN,
            "skewness": check_is_NaN,
            "sum": check_is_NaN,
            "range": check_is_NaN,
            "iqr": 18.0,
            "cv": check_is_NaN,
        },
        "y": {
            "n": "9",
            "n": 9,
            "count": 9,
            "p_missing": 0.0,
            "memory_size": None,
            "n_distinct": 9,
            "n_unique": 9,
            "p_distinct": 1.0,
            "is_unique": True,
            "p_unique": 1.0,
            "n_infinite": 0,
            "p_infinite": 0.0,
            "n_zeros": 0,
            "p_zeros": 0.0,
            "n_negative": 1,
            "p_negative": 0.1111111111111111,
            "monotonic": None,
            "quantiles": {
                "5%": -3.1415926535,
                "25%": 1e-06,
                "50%": 15.9,
                "75%": 111.0,
                "95%": 3122.0,
            },
            "mad": 15.9,
            "min": -3.1415926535,
            "max": check_is_NaN,
            "mean": check_is_NaN,
            "std": check_is_NaN,
            "variance": check_is_NaN,
            "kurtosis": check_is_NaN,
            "skewness": check_is_NaN,
            "sum": check_is_NaN,
            "range": check_is_NaN,
            "iqr": 110.999999,
            "cv": check_is_NaN,
            "chi_squared": None,
            "histogram": None,
        },
        "cat": {
            "n": 9,
            "count": 8,
            "p_missing": 0.1111111111111111,
            "memory_size": None,
            "n_distinct": 7,
            "n_unique": 6,
            "p_distinct": 0.875,
            "is_unique": False,
            "p_unique": 0.75,
            "chi_squared": None,
            "characters": None,
            "words": None,
            "length": None,
            "histogram_length": None,
        },
        "s1": {
            "n": 9,
            "count": 9,
            "p_missing": 0.0,
            "memory_size": None,
            "n_distinct": 1,
            "n_unique": 0,
            "p_distinct": 0.1111111111111111,
            "is_unique": False,
            "p_unique": 0.0,
            "n_infinite": 0,
            "p_infinite": 0.0,
            "n_zeros": 0,
            "p_zeros": 0.0,
            "n_negative": 0,
            "p_negative": 0.0,
            "monotonic": None,
            "quantiles": {"5%": 1.0, "25%": 1.0, "50%": 1.0, "75%": 1.0, "95%": 1.0},
            "mad": 0.0,
            "min": 1.0,
            "max": 1.0,
            "mean": 1.0,
            "std": 0.0,
            "variance": 0.0,
            "kurtosis": check_is_NaN,
            "skewness": check_is_NaN,
            "sum": 9.0,
            "range": 0.0,
            "iqr": 0.0,
            "cv": 0.0,
            "chi_squared": None,
            "histogram": None,
            "mode": 1.0,
        },
        "s2": {
            "count": 9,
            "cv": check_is_NaN,
            "n_distinct": 1,
            "histogram": check_is_NaN,
            "iqr": check_is_NaN,
            "is_unique": False,
            "kurtosis": check_is_NaN,
            "mad": check_is_NaN,
            "max": check_is_NaN,
            "mean": check_is_NaN,
            "min": check_is_NaN,
            "mini_histogram": check_is_NaN,
            "n_missing": 0,
            "p_missing": 0.0,
            "p_distinct": 0.1111111111111111,
            "p_zeros": check_is_NaN,
            "range": check_is_NaN,
            "skewness": check_is_NaN,
            "std": check_is_NaN,
            "sum": check_is_NaN,
            "variance": check_is_NaN,
        },
        "somedate": {
            "n": 9,
            "count": 8,
            "p_missing": 0.1111111111111111,
            "memory_size": None,
            "n_distinct": 6,
            "n_unique": 4,
            "p_distinct": 0.75,
            "is_unique": False,
            "p_unique": 0.5,
            "min": None,
            "max": None,
            "range": None,
            "chi_squared": None,
            "histogram": None,
        },
        "bool_tf": {
            "count": 9,
            "n_distinct": 2,
            "is_unique": False,
            "n_missing": 0,
            "p_missing": 0,
            "p_distinct": 2 / 9,
        },
        "bool_tf_with_nan": {
            "n": 9,
            "count": 9,
            "p_missing": 0.0,
            "memory_size": None,
            "n_distinct": 2,
            "n_unique": 0,
            "p_distinct": 0.2222222222222222,
            "is_unique": False,
            "p_unique": 0.0,
        },
        "bool_01": {
            "n": 9,
            "count": 9,
            "cv": 0.9486832980505138,
            "n_distinct": 2,
            "iqr": 1.0,
            "is_unique": False,
            "mad": 0,
            "max": 1,
            "min": 0,
            "n_missing": 0,
            "p_missing": 0,
            "p_distinct": 2 / 9,
            "p_zeros": 4 / 9,
            "sum": 5,
        },
        "bool_01_with_nan": {
            "n": 9,
            "count": 9,
            "p_missing": 0.0,
            "memory_size": None,
            "n_distinct": 3,
            "n_unique": 1,
            "p_distinct": 0.3333333333333333,
            "is_unique": False,
            "p_unique": 0.1111111111111111,
            "n_infinite": 0,
            "p_infinite": 0.0,
            "n_zeros": 4,
            "p_zeros": 0.4444444444444444,
            "n_negative": 0,
            "p_negative": 0.0,
            "monotonic": None,
            "quantiles": {"5%": 0.0, "25%": 0.0, "50%": 0.0, "75%": 1.0, "95%": 1.0},
            "mad": 0.0,
            "min": 0.0,
            "max": check_is_NaN,
            "mean": check_is_NaN,
            "std": check_is_NaN,
            "variance": check_is_NaN,
            "kurtosis": check_is_NaN,
            "skewness": check_is_NaN,
            "sum": check_is_NaN,
            "range": check_is_NaN,
            "iqr": 1.0,
            "cv": check_is_NaN,
            "chi_squared": None,
            "histogram": None,
        },
        "list": {
            "n": 9,
            "count": 9,
            "n_missing": 0,
            "p_missing": 0,
        },
        "mixed": {
            "n": 9,
            "count": 9,
            "n_missing": 0,
            "p_missing": 0,
        },
        "dict": {},
    }


@pytest.mark.sparktest
@pytest.mark.parametrize(
    "column",
    [
        "id",
        "x",
        "y",
        "cat",
        "s1",
        "s2",
        "somedate",
        "bool_tf",
        "bool_tf_with_nan",
        "bool_01",
        "bool_01_with_nan",
        "list",
        "mixed",
        "dict",
    ],
)
def test_describe_spark_df(
    column,
    describe_data,
    expected_results,
    summarizer,
    typeset,
    spark_session,
):

    cfg = SparkSettings()

    # disable correlations for description test
    cfg.correlations["pearson"].calculate = False
    cfg.correlations["spearman"].calculate = False

    if column == "mixed":
        describe_data[column] = [str(i) for i in describe_data[column]]
    if column == "bool_tf_with_nan":
        describe_data[column] = [True if i else False for i in describe_data[column]]
    sdf = spark_session.createDataFrame(pd.DataFrame({column: describe_data[column]}))

    results = describe(cfg, sdf, summarizer, typeset)

    assert {
        "analysis",
        "table",
        "variables",
        "scatter",
        "correlations",
        "missing",
        "package",
        "sample",
        "duplicates",
        "alerts",
    } == set(results.keys()), "Not in results"
    # Loop over variables
    for k, v in expected_results[column].items():
        if v == check_is_NaN:
            # test_condition should be True if column not in results, or the result is a nan value
            test_condition = k not in results["variables"][column] or np.isnan(
                results["variables"][column].get(k, np.NaN)
            )
        elif isinstance(v, float):
            test_condition = (
                pytest.approx(v, nan_ok=True) == results["variables"][column][k]
            )
        else:
            test_condition = v == results["variables"][column][k]

        assert (
            test_condition
        ), f"Value `{results['variables'][column][k]}` for key `{k}` in column `{column}` is not check_is_NaN"
