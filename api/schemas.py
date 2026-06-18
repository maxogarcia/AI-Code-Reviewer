from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    code: str = Field(..., description="Code snippet to review", min_length=1)
    top_k: int = Field(3, description="Number of RAG examples to retrieve", ge=1, le=10)
    max_new_tokens: int = Field(512, description="Max tokens for the generated review", ge=64, le=2048)


class RetrievedExample(BaseModel):
    code: str
    review: str
    distance: float


class ReviewResponse(BaseModel):
    review: str
    retrieved_examples: list[RetrievedExample]
