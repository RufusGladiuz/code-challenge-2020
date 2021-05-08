from dataclasses import dataclass

# TODO: Add Comments


@dataclass
class QuantileCutOrder:
    """Simple data class to hold setting for a categorical quantile cut

        Args:
            outlier_column: str -> Column to apply the quantile action to.
            per_category: str -> Category to consider when apply the quantile action.
            quantile: float -> A float between 0 and 1 determining the quantile cut.
    """
    outlier_column: str
    per_category: str
    quantile: float
