import unittest
import indexing_process
from documents import Document, DictDocumentStore
from query_process import QueryProcess
from tf_idf_inverted_index import TfIdfInvertedIndex


class TestAdvancedSearch(unittest.TestCase):
    def setUp(self):
        # Create a document store and index for testing
        self.document_store = DictDocumentStore()
        self.index = TfIdfInvertedIndex()

        self.testDocuments = [
            Document(doc_id='0', text='red is a color'),
            Document(doc_id='1', text='red and blue'),
            Document(doc_id='2', text='green and red colors'),
        ]

        for doc in self.testDocuments:
            self.document_store.add_document(doc)

        transformed_documents = indexing_process.transform_documents(self.testDocuments)
        for document in transformed_documents:
            self.index.add_document(document)

        self.query_processor = QueryProcess(self.document_store, self.index)

    def test_regular_term_search(self):
        # Test searching for a regular term
        query = '"and"'
        result = self.query_processor.search(query, number_of_results=10)
        self.assertNotIn("(0) red is a color", result)
        self.assertIn("(1) red and blue", result)
        self.assertIn("(2) green and red colors", result)

    def test_phrase_search(self):
        # Test searching for a phrase enclosed in double quotes
        query = '"is a color"'
        result = self.query_processor.search(query, number_of_results=10)
        self.assertIn("(0) red is a color", result)
        self.assertNotIn("(1) red and blue", result)
        self.assertNotIn("(2) green and red colors", result)

    def test_phrase_search_with_different_order(self):
        # Test searching for a phrase with different word order
        query = '"blue and red"'
        result = self.query_processor.search(query, number_of_results=10)
        self.assertNotIn("(0) red is a color", result)
        self.assertIn("(1) red and blue", result)
        self.assertNotIn("(2) green and red colors", result)

    def test_phrase_search_with_nonexistent_phrase(self):
        # Test searching for a phrase that doesn't exist in any documents
        query = '"yellow and orange"'
        result = self.query_processor.search(query, number_of_results=10)
        self.assertNotIn("(0) red is a color", result)
        self.assertNotIn("(1) red and blue", result)
        self.assertNotIn("(2) green and red colors", result)

    def test_mixed_search(self):
        # Test searching for a phrase enclosed in double quotes
        query = 'colors "green and red"'
        result = self.query_processor.search(query, number_of_results=10)
        self.assertNotIn("(0) red is a color", result)
        self.assertNotIn("(1) red and blue", result)
        self.assertIn("(2) green and red colors", result)

    def test_multi_phrase_search(self):
        # Test searching for multiple phrase enclosed in double quotes
        query = '"a color" "red is"'
        result = self.query_processor.search(query, number_of_results=10)
        self.assertIn("(0) red is a color", result)
        self.assertNotIn("(1) red and blue", result)
        self.assertNotIn("(2) green and red colors", result)


if __name__ == '__main__':
    unittest.main()
