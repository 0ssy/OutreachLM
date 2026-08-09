from pathlib import Path
from datasets import load_dataset

def download_fineweb(
    output_directory="corpus/text",
    number_of_documents=1000,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="CC-MAIN-2024-10",
        split="train",
        streaming=True
    )

    saved=0

    for document in dataset:
        text = document["text"]
        if not text.strip():
            continue

        output_file =output_directory / f"fineweb_{saved:05d}.txt"
        with open(
            output_file,
            "w",
            encoding="utf-8"
        
        ) as file:
            file.write(text)

        saved += 1

        print(f"Saved Document {saved}/{number_of_documents}")

        if saved >= number_of_documents:
            break
        print()
        print("finished downloading fineweb dataset")
        print("Documents saved:", saved)
if __name__ == "__main__":
    download_fineweb(
        output_directory="corpus/text",
        number_of_documents=1000
    )