import time
import json
import spacy
import redis
from nltk.corpus import stopwords

redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)


class SpacyTagger(object):

    def __init__(self, lang, disable=[]):
        self.nlp = spacy.load(self.get_lang_resource(lang), disable=disable)
        self.set_stopwords(lang)

    @staticmethod
    def get_lang_resource(lang):
        valid_langs = {
            "en": "en_core_web_sm",
            "de": "de_core_news_sm"
        }
        return valid_langs.get(lang, 'en_core_web_sm')

    def set_stopwords(self, lang):
        if lang == 'en':
            self.stops = stopwords.words('english')
        if lang == 'de':
            self.stops = stopwords.words('german')
        else:
            self.stops = []

    def get_nouns(self, doc):
        return [t.text for t in self.nlp(doc) if t.pos_ == 'NOUN']

    def get_noun_chunks(self, doc):
        return [" ".join([t.text for t in nc if t.is_stop is False])
                for nc in self.nlp(doc).noun_chunks if len(nc) > 0]

    def get_noun_chunks_batch(self, docs):
        noun_chunks = [[" ".join([t.text for t in nc if (
                                        t.is_stop is False
                                        and t.text != ""
                                        and t.text.lower() not in self.stops)])
                        for nc in d.noun_chunks if len(nc) > 0]
                       for d in self.nlp.pipe(docs)]
        noun_chunks = [nc for nc in noun_chunks if nc != ""]
        return noun_chunks

    def get_entities_batch(self, docs):
        entities = [[str(e) for e in d.ents if str(e) != ""]
                    for d in self.nlp.pipe(docs)]
        return entities

    def get_tokens_batch(self, docs):
        tokens = [[t.text for t in d if t.is_alpha]
                  for d in self.nlp.pipe(docs, batch_size=100, n_threads=5)]
        return tokens

    def get_sentences_batch(self, docs):
        return [[" ".join([t.text.lower() for t in s if t.is_alpha])
                 for s in doc.sents]
                for doc in self.nlp.pipe(docs)]


def run_pos_process(lang):
    tagger = SpacyTagger(lang, ['ner'])
    while True:
        q = redis_store.lpop('pos_'+lang)
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            doc = d["doc"]
            nouns = tagger.get_nouns(doc)
            noun_chunks = tagger.get_noun_chunks(doc)
            result = {"nouns": nouns,
                      "noun_chunks": noun_chunks}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)


def run_pos_process_batch(lang):
    tagger = SpacyTagger(lang, ['ner'])
    while True:
        q = redis_store.lpop('batch_pos_'+lang)
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            if not type(d["docs"]) == list:
                docs = json.loads(d["docs"])
            else:
                docs = d["docs"]
            noun_chunks = tagger.get_noun_chunks_batch(docs)
            result = {"noun_chunks": noun_chunks}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)


def run_ner_process_batch(lang):
    tagger = SpacyTagger(lang)
    while True:
        q = redis_store.lpop('batch_ner_'+lang)
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            if not type(d["docs"]) == list:
                docs = json.loads(d["docs"])
            else:
                docs = d["docs"]
            entities = tagger.get_entities_batch(docs)
            result = {"entities": entities}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)


def run_tokenize_process_batch(lang):
    tokenizer = SpacyTagger(lang, disable=['tagger', 'parser', 'ner'])
    while True:
        q = redis_store.lpop('batch_tokenize_'+lang)
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            if not type(d["docs"]) == list:
                docs = json.loads(d["docs"])
            else:
                docs = d["docs"]
            entities = tokenizer.get_tokens_batch(docs)
            result = {"tokens": entities}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)
