import unittest

from arxiv_paper_chatbot.parser import PaperParser
from py_pdf_parser.loaders import PDFDocument, load_file
from arxiv_paper_chatbot.paper import Paper

import unittest


class TestParseTitle(unittest.TestCase):
    def setUp(self):
        self.parser = PaperParser()

    def test_parse_with_empty_document(self):
        doc = PDFDocument(pages={})
        result = self.parser.parse(doc)
        self.assertIsNone(result)

    def test_parse_with_valid_document(self):
        doc = load_file("tests/resources/test.pdf")
        result = self.parser.parse(doc)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Paper)
        self.assertTrue("Lorem ipsum" in result.raw)
