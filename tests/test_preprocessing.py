
import pandas as pd


def test_missing_values():
    data = pd.DataFrame({
        "TotalCharges": ["100", "200", ""]
    })

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    assert data["TotalCharges"].isna().sum() == 1


def test_numeric_conversion():
    data = pd.Series(["100", "200", "300"])

    result = pd.to_numeric(data, errors="coerce")

    assert result.dtype in ["int64", "float64"]
    assert result.notna().all()
