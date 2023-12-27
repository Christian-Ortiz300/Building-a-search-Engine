import json
import math
from collections import defaultdict, Counter

from documents import TransformedDocument
from index import BaseIndex


def count_terms(terms: list[str]) -> Counter:
    return Counter(terms)


class TfIdfInvertedIndex(BaseIndex):
    def __init__(self, total_documents_count=0):
        # Mapping of terms to the number of documents they occur in.
        self.doc_counts = Counter()
        self.total_documents_count = total_documents_count
        self.term_to_doc_id_tf_scores = defaultdict(dict)
        self.term_to_doc_id_tf_positions = defaultdict(dict)

    def write(self, path: str):
        with open(path, 'w') as fp:
            fp.write(json.dumps({
                '__metadata__': {
                    'doc_counts': [
                        {
                            'term': term,
                            'count': count
                        }
                        for term, count in self.doc_counts.items()
                    ]
                }
            }) + '\n')
            for doc_id, term_tf_scores in self.term_to_doc_id_tf_scores.items():
                fp.write(json.dumps({
                    'doc_id': doc_id,
                    'term_tf_scores': [
                        {
                            'term': term,
                            'tf_score': tf_score
                        }
                        for term, tf_score in term_tf_scores.items()
                    ]
                }) + '\n')

    def add_document(self, doc: TransformedDocument):
        term_counts = count_terms(doc.terms)
        self.doc_counts.update(term_counts.keys())
        self.total_documents_count += 1

        for position, term in enumerate(doc.terms):
            self.term_to_doc_id_tf_scores[term][doc.doc_id] = term_counts[term] / len(doc.terms)

            # Ensure the initialization of term_to_doc_id_tf_positions for each term
            if term not in self.term_to_doc_id_tf_positions:
                self.term_to_doc_id_tf_positions[term] = {}

            if doc.doc_id not in self.term_to_doc_id_tf_positions[term]:
                self.term_to_doc_id_tf_positions[term][doc.doc_id] = []

            self.term_to_doc_id_tf_positions[term][doc.doc_id].append(position)

    def term_frequency(self, term, doc_id):
        return self.term_to_doc_id_tf_scores[term].get(doc_id, 0.0)

    def inverse_document_frequency(self, term):
        return math.log(self.total_documents_count / self.doc_counts[term])

    def tf_idf(self, term, doc_id):
        return self.term_frequency(term, doc_id) * self.inverse_document_frequency(term)

    def combine_term_scores(self, terms: list[str], query_doc_id: str) -> float:
        matching_doc_ids = set(self.term_to_doc_id_tf_scores[terms[0]].keys())
        for term in terms:
            matching_doc_ids.intersection_update(self.term_to_doc_id_tf_scores[term].keys())

        score = 0.0
        for doc_id in matching_doc_ids:
            if doc_id == query_doc_id and self.do_terms_form_valid_phrase(terms, doc_id):
                score += self.calculate_phrase_score(terms, doc_id)
        return score

    def do_terms_form_valid_phrase(self, terms, doc_id):
        # Check if the terms in the query form a valid phrase in the document
        positions = [self.term_to_doc_id_tf_positions[term][doc_id] for term in terms]
        # Check if the positions are consecutive
        return all(pos in positions[i + 1] for i, pos in enumerate(positions[:-1]))

    def calculate_phrase_score(self, terms, doc_id):
        # Calculate the score for a valid phrase
        return sum([self.tf_idf(term, doc_id) for term in terms])

    def search(self, processed_query: list[str], number_of_results: int) -> list[str]:
        matching_doc_ids = set(self.term_to_doc_id_tf_scores[processed_query[0]].keys())
        for term in processed_query:
            matching_doc_ids.intersection_update(self.term_to_doc_id_tf_scores[term].keys())

        # Filter matching_doc_ids to include only those documents where the terms form a valid phrase
        # matching_doc_ids = [doc_id for doc_id in matching_doc_ids if
        #                     self.do_terms_form_valid_phrase(processed_query, doc_id)]

        scores = {doc_id: self.combine_term_scores(processed_query, doc_id) for doc_id in matching_doc_ids}
        return sorted(matching_doc_ids, key=scores.get, reverse=True)[:number_of_results]

    # def search(self, processed_query: list[str], number_of_results: int) -> list[str]:
    #     terms = []
    #     phrases = []
    #     for item in processed_query:
    #         if ' ' in item:  # If space exists, it's a phrase
    #             phrases.append(item)
    #         else:
    #             terms.append(item)
    #
    #     matching_doc_ids = set(self.term_to_doc_id_tf_scores[terms[0]].keys())
    #     for term in terms:
    #         matching_doc_ids.intersection_update(self.term_to_doc_id_tf_scores[term].keys())
    #
    #     matching_doc_ids = [doc_id for doc_id in matching_doc_ids if self.do_terms_form_valid_phrase(terms, doc_id)]
    #
    #     phrase_scores = {doc_id: self.combine_term_scores(phrase.split(), doc_id) for phrase in phrases}
    #
    #     scores = {doc_id: self.combine_term_scores(terms, doc_id) + score for doc_id, score in phrase_scores.items()}
    #
    #     return sorted(matching_doc_ids, key=scores.get, reverse=True)[:number_of_results]