from dataclasses import dataclass

#TODO: Add Comments

@dataclass
class QuantileCutOrder:
    outlier_column:str
    per_category:str
    quantile:str 
