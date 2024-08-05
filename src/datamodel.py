from pydantic import BaseModel, Field

class Data(BaseModel):
    no_of_dependents: int = None
    education: int = Field(default=None, ge=0, le=1)
    self_employed: int = Field(default=None, ge=0, le=1)
    income_annum: int = None
    loan_amount: int = None
    loan_term: int = None
    cibil_score: int = None
    residential_assets_value: int = None
    commercial_assets_value: int = None
    luxury_assets_value: int = None
    bank_asset_value: int = None
