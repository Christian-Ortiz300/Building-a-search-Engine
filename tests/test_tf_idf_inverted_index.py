import unittest
from unittest import TestCase

from tf_idf_inverted_index import TfIdfInvertedIndex, TransformedDocument


class TestTfIdfInvertedIndexAddDocument(unittest.TestCase):
    def test_add_document(self):
        index = TfIdfInvertedIndex()
        doc1 = TransformedDocument(doc_id="doc1", terms=["apple", "banana", "cherry"])
        doc2 = TransformedDocument(doc_id="doc2", terms=["banana", "date", "elderberry"])

        index.add_document(doc1)
        index.add_document(doc2)

        # Ensure that terms are correctly added to the index
        self.assertIn("apple", index.term_to_doc_id_tf_scores)
        self.assertIn("banana", index.term_to_doc_id_tf_scores)
        self.assertIn("cherry", index.term_to_doc_id_tf_scores)
        self.assertIn("date", index.term_to_doc_id_tf_scores)
        self.assertIn("elderberry", index.term_to_doc_id_tf_scores)

        # Ensure that the document counts are updated
        self.assertEqual(index.total_documents_count, 2)

        # Ensure that the document term frequencies are correctly calculated
        self.assertEqual(index.term_frequency("apple", "doc1"), 1.0 / 3)
        self.assertEqual(index.term_frequency("banana", "doc1"), 1.0 / 3)
        self.assertEqual(index.term_frequency("cherry", "doc1"), 1.0 / 3)

        self.assertEqual(index.term_frequency("banana", "doc2"), 1.0 / 3)
        self.assertEqual(index.term_frequency("date", "doc2"), 1.0 / 3)
        self.assertEqual(index.term_frequency("elderberry", "doc2"), 1.0 / 3)


if __name__ == '__main__':
    unittest.main()


class TestTfIdfInvertedIndex(TestCase):
    def test_search(self):
        self.fail()
