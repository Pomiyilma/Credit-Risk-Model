from pydantic import BaseModel, Field
from typing import List

class CreditRiskRequest(BaseModel):
    # Core numerical inputs used by our clean model features
    Amount: float = Field(..., description="The transaction amount", example=5000.0)
    Value: float = Field(..., description="The value of the transaction", example=5000.0)
    PricingStrategy: int = Field(..., description="The pricing strategy category integer", example=2)
    
    # Flags for categorical items after dummy encoding (defaulted to 0 or False)
    ChannelId_ChannelId_2: int = Field(0, example=0)
    ChannelId_ChannelId_3: int = Field(0, example=0)
    ChannelId_ChannelId_5: int = Field(0, example=0)
    ProviderId_ProviderId_4: int = Field(0, example=0)
    ProductCategory_financial_services: int = Field(0, example=1)
    ProductId_ProductId_6: int = Field(0, example=0)
    ProductId_ProductId_10: int = Field(0, example=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Amount": 1200.50,
                "Value": 1200.50,
                "PricingStrategy": 2,
                "ChannelId_ChannelId_2": 1,
                "ChannelId_ChannelId_3": 0,
                "ChannelId_ChannelId_5": 0,
                "ProviderId_ProviderId_4": 1,
                "ProductCategory_financial_services": 1,
                "ProductId_ProductId_6": 0,
                "ProductId_ProductId_10": 0
            }
        }

class CreditRiskResponse(BaseModel):
    risk_probability: float = Field(..., description="Probability of high risk (0.0 to 1.0)")
    is_high_risk: bool = Field(..., description="True if risk probability exceeds 0.5")
    