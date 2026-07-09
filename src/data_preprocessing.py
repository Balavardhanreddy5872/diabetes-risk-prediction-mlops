from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def build_preprocessor():
    """
    Builds preprocessing pipeline for the diabetes risk prediction dataset.

    Continuous features are scaled.
    Remaining numeric/binary/ordinal features are passed through unchanged.
    """

    continuous_features = ["BMI", "MentHlth", "PhysHlth"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous_scaler", StandardScaler(), continuous_features)
        ],
        remainder="passthrough"
    )

    return preprocessor