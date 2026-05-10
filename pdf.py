#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter


def collect_pdfs(folder: Path, output_file: Path):
    pdfs = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() == ".pdf":
            if f.resolve() != output_file.resolve():
                pdfs.append(f)
    return pdfs


def main():
    parser = argparse.ArgumentParser(
        description="Combine the first 2 pages of every PDF in a folder into one PDF."
    )
    parser.add_argument(
        "folder",
        help="Folder containing input PDF files"
    )
    parser.add_argument(
        "-o", "--output",
        default="combined_first_2_pages.pdf",
        help="Output PDF filename (default: combined_first_2_pages.pdf)"
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder}' is not a valid folder.", file=sys.stderr)
        sys.exit(1)

    pdf_files = collect_pdfs(folder, output_file)

    if not pdf_files:
        print("No PDF files found.")
        sys.exit(1)

    writer = PdfWriter()
    processed = 0

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    print(f"Skipping encrypted PDF: {pdf_path.name}")
                    continue

            pages_to_take = min(2, len(reader.pages))
            if pages_to_take == 0:
                print(f"Skipping empty PDF: {pdf_path.name}")
                continue

            for i in range(pages_to_take):
                writer.add_page(reader.pages[i])

            processed += 1
            print(f"Added {pages_to_take} page(s) from: {pdf_path.name}")

        except Exception as e:
            print(f"Skipping '{pdf_path.name}' due to error: {e}")

    if len(writer.pages) == 0:
        print("No pages were added. Output PDF was not created.")
        sys.exit(1)

    with open(output_file, "wb") as f:
        writer.write(f)

    print(f"\nCreated: {output_file}")
    print(f"Processed PDF files: {processed}")


if __name__ == "__main__":
    main()