import os

from dotenv import load_dotenv
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel


class DocumentType(str, Enum):
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    RECEIPT = "receipt"
    OTHER = "other"


class DocumentClassification(BaseModel):
    document_type: DocumentType
    language: str
    requires_review: bool
    

def main():
    load_dotenv()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    document = """
    Hello,

    Please find the attached information.
    We discussed the amount yesterday.

    Best regards,
    Alex
    """

    response = client.responses.parse(
        model="gpt-5-mini",
        instructions=(
             "You classify business documents into invoice, purchase_order, "
            "receipt, or other. "
            "Use the language's ISO 639-1 code, such as 'en'. "
            "Classify as 'other' when the document is clearly a different "
            "document type or when its type cannot be determined. "
            "Set requires_review to true when the text is incomplete, "
            "references missing attachments, or does not contain enough "
            "information to reliably identify its document type. "
            "A classification of 'other' does not automatically mean "
            "requires_review is false. "
            "Set requires_review to false for clearly identifiable documents, "
            "including documents that clearly belong to the 'other' category."
        ),
        input=document,
        text_format=DocumentClassification,
        max_output_tokens=500,
    )

    if response.status != "completed":
        raise RuntimeError(
            f"Incomplete response: {response.incomplete_details}"
        )

    classification = response.output_parsed

    if classification is None:
        raise RuntimeError("No parsed classification returned")

    print(classification.model_dump_json(indent=2))
    print("Token usage:", response.usage.total_tokens)


if __name__ == "__main__":
    main()