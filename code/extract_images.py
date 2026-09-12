#!/usr/bin/env python3
"""Prepare OCR review records for new images. Never guess a financial amount.

Optional dependencies: Pillow, pytesseract and the system tesseract executable.
The checked-in image cache already covers the supplied dataset; OCR is optional.
"""
import argparse
import hashlib
import json
from pathlib import Path
from common import ROOT, rows


def extract(dataset, out):
    import pytesseract
    from PIL import Image, ImageOps
    out.mkdir(parents=True, exist_ok=True)
    for link in rows(dataset / 'images.csv'):
        image_id = link['image_id']
        if Path(image_id).name != image_id:
            raise ValueError('Invalid image ID')
        path = dataset / 'media/images' / (image_id + '.png')
        if not path.is_file():
            raise ValueError(f'Missing image {image_id}')
        with Image.open(path) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
            # Document layout varies; collect both automatic and sparse-text OCR.
            texts = {str(psm): pytesseract.image_to_string(im, config=f'--psm {psm}') for psm in (3, 11)}
        record = {**link, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'amount': None, 'currency': None, 'confidence': 'needs_review',
                  'source_quote': '', 'reason': '', 'ocr_text': texts}
        (out / (image_id + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(f'OCR review records written to {out}. Review totals against images before merging into image_facts.json.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'cache/ocr')
    a = p.parse_args()
    extract(a.dataset, a.out)
