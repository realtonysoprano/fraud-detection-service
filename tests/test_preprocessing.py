import pandas as pd

from src.preprocessing import preprocess_transactions


def test_preprocess_transactions_returns_dataframe():
    df = pd.DataFrame(
        [
            {
                "TransactionID": 1,
                "TransactionAmt": 100.0,
                "ProductCD": "W",
            }
        ]
    )

    result = preprocess_transactions(df)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert result.isna().sum().sum() == 0