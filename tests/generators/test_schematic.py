import pandas as pd
import pytest

from crnsynth.generators.schematic import SchematicGenerator

DATE_FORMAT = "%d-%m-%Y"


@pytest.fixture
def data_real():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [1.1, 2.2, 3.3, 4.4, 5.5],
            "D": pd.to_datetime(
                ["01-01-2020", "02-01-2020", "03-01-2020", "04-01-2020", "05-01-2020"],
                format=DATE_FORMAT,
            ),
        }
    )


def test_fit(data_real):
    generator = SchematicGenerator(date_format=DATE_FORMAT)
    generator.fit(data_real)
    assert generator.schema["A"] == {"dtype": "int", "min": 1, "max": 5}
    assert generator.schema["B"] == {
        "dtype": "object",
        "categories": ["a", "b", "c", "d", "e"],
    }
    assert generator.schema["C"] == {"dtype": "float", "min": 1.1, "max": 5.5}
    assert generator.schema["D"] == {
        "dtype": "datetime64[ns]",
        "min": pd.Timestamp("2020-01-1"),
        "max": pd.Timestamp("2020-01-05"),
    }


def test_generate(data_real):
    generator = SchematicGenerator(date_format=DATE_FORMAT)
    generator.fit(data_real)
    synthetic_data = generator.generate(10)
    assert synthetic_data.shape == (10, 4)
    assert synthetic_data["A"].dtype == "int"
    assert synthetic_data["B"].dtype == "object"
    assert synthetic_data["C"].dtype == "float"
    assert synthetic_data["D"].dtype == "datetime64[ns]"

    assert all(synthetic_data["A"].isin([1, 2, 3, 4, 5]))
    assert all(synthetic_data["B"].isin(["a", "b", "c", "d", "e"]))
    assert all(synthetic_data["C"].between(1.1, 5.5))
    assert all(
        synthetic_data["D"].between(
            pd.Timestamp("01-01-2020"), pd.Timestamp("05-01-2020")
        )
    )
