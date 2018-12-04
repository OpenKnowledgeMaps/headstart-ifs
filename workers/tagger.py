import time
import json
import spacy
import redis

redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)


class SpacyTagger(object):

    def __init__(self, lang, disable=[]):
        self.nlp = spacy.load(self.get_lang_resource(lang), disable=disable)

    @staticmethod
    def get_lang_resource(language):
        valid_langs = {
            "english": "en_core_web_sm",
            "german": "de_core_news_sm"
        }
        return valid_langs.get(language, 'en_core_web_sm')

    def get_nouns(self, doc):
        doc = self.nlp(doc)
        return [str(t) for t in doc if t.pos_ == 'NOUN']

    def get_noun_chunks(self, doc):
        doc = self.nlp(doc)
        return [" ".join([str(t) for t in nc if t.is_stop is False])
                for nc in doc.noun_chunks if len(nc) > 0]

    def get_noun_chunks_batch(self, docs):
        noun_chunks = [[" ".join([str(t) for t in nc if t.is_stop is False])
                        for nc in d.noun_chunks if len(nc) > 0]
                       for d in self.nlp.pipe(docs)]
        return noun_chunks

    def get_entities_batch(self, docs):
        entities = [[str(e) for e in d.ents]
                    for d in self.nlp.pipe(docs)]
        return entities


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
            docs = json.loads(d["docs"])
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
            docs = json.loads(d["docs"])
            entities = tagger.get_entities_batch(docs)
            result = {"entities": entities}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)
