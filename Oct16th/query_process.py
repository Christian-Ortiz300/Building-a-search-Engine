import json
from documents import DictDocumentStore
from tf_idf_inverted_index import BaseIndex
import tokenizer


def preprocess_query(query: str):
    query.split()


class QueryProcess:
    def __init__(self, document_store: DictDocumentStore, index: BaseIndex, stopwords_file=None):
        self.document_store = document_store
        self.index = index
        self.stopwords = self.load_stopwords(stopwords_file) if stopwords_file else set()

    @staticmethod
    def load_stopwords(stopwords_file):
        if stopwords_file:
            try:
                with open(stopwords_file, 'r') as fp:
                    stopwords = json.load(fp)
                    return set(stopwords)
            except FileNotFoundError:
                print("Stopwords file not found. Using an empty stopwords list.")
            except json.JSONDecodeError:
                print("Error parsing JSON in stopwords file. Using an empty stopwords list.")
            return set()

    def remove_stopwords(self, query_terms):
        gowords = [term for term in query_terms if term not in self.stopwords]
        return gowords

    def search(self, query: str, number_of_results: int) -> str:
        parsed_query = self.parse_query(query)
        results = self.index.search(parsed_query, number_of_results)
        return self.format_out(results, self.document_store, parsed_query)

    def parse_query(self, query: str):
        # Split the query into terms and detect phrases enclosed in double quotes
        terms_and_phrases = self.split_query(query)
        return terms_and_phrases

    def split_query(self, query: str) -> list[str]:
        terms_and_phrases = []
        current_phrase= []
        in_phrase = False
        querysplit = query.split()

        for term in querysplit:
            if term.startswith('"'):
                # Handle a complete phrase enclosed in double quotes
                current_phrase.append(term.strip('"'))
                in_phrase = True
            elif term.endswith('"'):
                # Start of a new phrase
                current_phrase.append(term.strip('"'))
            elif in_phrase:
                # Continuing a phrase
                current_phrase.append(term)
            elif not in_phrase:
                # A regular term
                terms_and_phrases.append(term)
        if current_phrase:
            return current_phrase
        else:
            return terms_and_phrases

    def format_out(self, results: list[str], document_store: DictDocumentStore, unused_processed_query) -> str:
        output_string = ''
        for doc_id in results:
            doc = document_store.get_by_doc_id(doc_id)
            if doc:
                output_string += f'({doc.doc_id}) {doc.text}\n\n'
        return output_string
