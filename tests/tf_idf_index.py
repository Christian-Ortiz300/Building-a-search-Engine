import json
import math
from collections import defaultdict, Counter

from documents import TransformedDocument
from index import BaseIndex


def count_terms(terms: list[str]) -> Counter:
    return Counter(terms)


class TfIdfIndex(BaseIndex):
    def __init__(self):
        # Mapping of terms to the number of documents they occur in.
        self.doc_counts = Counter()
        self.term_to_doc_id_tf_scores: dict[str, Counter] = dict()

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
            for doc_id, counts in self.term_to_doc_id_tf_scores.items():
                fp.write(json.dumps({
                    'doc_id': doc_id,
                    'counts': [
                        {
                            'term': term,
                            'count': count
                        }
                        for term, count in counts.items()
                    ]
                }) + '\n')

    def add_document(self, doc: TransformedDocument):
        term_counts = count_terms(doc.terms)
        self.doc_counts.update(term_counts.keys())

        # Mapping from doc_ids to term counts in the corresponding document.
        self.term_to_doc_id_tf_scores[doc.doc_id] = term_counts

    def term_frequency(self, term, doc_id):
        return self.term_to_doc_id_tf_scores[doc_id][term] / self.term_to_doc_id_tf_scores[doc_id].total()

    def inverse_document_frequency(self, term):
        term_count = self.doc_counts[term]
        if term_count == 0:
            return 0.0
        return math.log(len(self.term_to_doc_id_tf_scores) / self.doc_counts[term])

    def tf_idf(self, term, doc_id):
        return self.term_frequency(term, doc_id) * self.inverse_document_frequency(term)

    def combine_term_scores(self, terms: list[str], doc_id) -> float:
        return sum([self.tf_idf(term, doc_id) for term in terms])

    def search(self, processed_query: list[str], number_of_results: int) -> list[str]:
        scores = dict()
        for doc_id, counts in self.term_to_doc_id_tf_scores.items():
            score = self.combine_term_scores(processed_query, doc_id)
            scores[doc_id] = score
        return sorted(self.term_to_doc_id_tf_scores.keys(), key=scores.get, reverse=True)[:number_of_results]


