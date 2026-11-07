from pydantic import BaseModel, Field, field_validator


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, examples=["Hello! How are you today?"])

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("o texto não pode conter apenas espaços")
        return value


class TranslationResponse(BaseModel):
    source_language: str = "en"
    target_language: str = "fr"
    original_text: str
    translated_text: str
    model: str
    elapsed_ms: float
