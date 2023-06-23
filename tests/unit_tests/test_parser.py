import unittest

import pytest
from typing import List

from arxiv_paper_chatbot.parser import PaperParser  # Import the class containing _parse_title
from py_pdf_parser.loaders import PDFDocument, load_file


class TestParseTitle(unittest.TestCase):
    def test_parse_title_with_valid_data(self):
        doc = PDFDocument(pages={}) # TODO
        result = PaperParser._parse_title(doc)
        self.assertEqual(result, 'This is a title')

    def test_parse_title_with_empty_data(self):
        empty_doc = PDFDocument({})
        result = PaperParser._parse_title(empty_doc)
        self.assertEqual(result, "")

    def test_parse_title_with_no_valid_title(self):
        # Create a PDFDocument with no valid title
        sample_data = {
            'pages': [
                {
                    'elements': [
                        {'text': 'Short text', 'font_size': 8},
                        {'text': 'Another short text', 'font_size': 9},
                    ]
                }
            ]
        }
        doc = PDFDocument(pages={}) # TODO

        # Call the _parse_title method and assert the result
        result = PaperParser._parse_title(doc)
        self.assertEqual(result, "")