from markthat import MarkThat as MarkThatNew
from dotenv import load_dotenv
import os, asyncio

load_dotenv()

def test_markthat_with_figure_extraction():
    try:
        client = MarkThatNew(
            provider="gemini",
            model="gemini-2.0-flash-001",
            api_key=os.getenv("GEMINI_API_KEY"),
            api_key_figure_detector=os.getenv("GEMINI_API_KEY"),
            api_key_figure_extractor=os.getenv("GEMINI_API_KEY"),
            api_key_figure_parser=os.getenv("GEMINI_API_KEY"),
        )

        result = asyncio.run(
            client.async_convert(
                "/Users/davidperso/projects/markthat/2507.23726v1-1.pdf",
                extract_figure=True,
                coordinate_model="gemini-2.0-flash-001",
                parsing_model="gemini-2.5-flash-lite",
            )
        )
        return result
    except Exception as e:
        print("New client failed with error: ", e)

def test_markthat_without_figure_extraction():
    try:
        client = MarkThatNew(
            provider="gemini",
            model="gemini-2.0-flash-001",
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        result = asyncio.run(
            client.async_convert(
                "/Users/davidperso/projects/markthat/2507.23726v1-1.pdf",
                extract_figure=False,
            )
        )
        return result
    except Exception as e:
        print("New client failed with error: ", e)

if __name__ == "__main__":
    markthat_with_figure_extraction_result = test_markthat_with_figure_extraction()
    markthat_without_figure_extraction_result = test_markthat_without_figure_extraction()
    print("\n\n\n\n\n\n\n")
    print("Markthat with figure extraction result: ", markthat_with_figure_extraction_result)
    print("\n\n\n\n\n\n\n")
    print("Markthat without figure extraction result: ", markthat_without_figure_extraction_result)
    
    
