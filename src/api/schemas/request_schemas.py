from pydantic import BaseModel

class PredictionRequest(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    currency_code: str
    product_category: str
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
