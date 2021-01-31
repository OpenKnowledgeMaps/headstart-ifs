import sys
import logging
import time
import json
import spacy

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
core_models = {"zh": "web",
               "da": "news",
               "nl": "news",
               "en": "web",
               "fr": "news",
               "de": "news",
               "el": "news",
               "it": "news",
               "ja": "news",
               "lt": "news",
               "pl": "news",
               "pt": "news",
               "ro": "news",
               "es": "news"}
                              
class SpacyNLP(object):

    def __init__(self, redis_store, loglevel):
        self.redis_store = redis_store
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(loglevel)
        self.logger.addHandler(handler)
        self.load_models()

    def load_models(self):
        self.pipes = {
            "sent_tokenize": {},
            "nlp": {}
        }
        for cl, source in core_models.items():
            self.logger.debug("Loading model: %s" % cl)
            self.pipes["sent_tokenize"][cl] = spacy.load("%s_core_%s_sm" % (cl, source),
                                        disable=["ner", "entity_linker",
                                                 "textcat", "entity_ruler",
                                                 "merge_noun_chunks",
                                                 "merge_entities",
                                                 "merge_subtokens"])
            self.pipes["nlp"][cl] = spacy.load("%s_core_%s_sm" % (cl, source),
                                        disable=["textcat", "entity_ruler",
                                                 "merge_subtokens"])
        self.logger.debug("Models loaded")

    def next_item(self):
        queue, msg = self.redis_store.blpop("nlp")
        msg = json.loads(msg)
        k = msg.get('id')
        docs = msg.get('docs')
        lang = msg.get('lang')
        tasks = msg.get('tasks')
        return k, docs, lang, tasks

    def sent_tokenize(self, docs, lang):
        assert isinstance(docs, list)
        for d in docs:
            assert isinstance(d, str), "assert failed: isinstance(doc, str)"
        if lang in core_models:
            nlp = self.pipes["sent_tokenize"][lang]
        else:
            nlp = self.pipes["sent_tokenize"]["en"]
        tokenized_docs = [d for d in nlp.pipe(docs, batch_size=100, n_threads=5)]
        sents = [[s.text for s in d.sents] for d in tokenized_docs]
        return sents

    def tokenize_docs(self, docs, lang):
        assert isinstance(docs, list)
        for d in docs:
            assert isinstance(d, str), "assert failed: isinstance(doc, str)"
        if lang in core_models:
            nlp = self.pipes["nlp"][lang]
        else:
            nlp = self.pipes["nlp"]["en"]
        tokenized_docs = [d for d in nlp.pipe(docs, batch_size=100, n_threads=5)]
        return tokenized_docs

    def nlp(self, docs, lang, tasks):
        res = {}
        tokenized_docs = self.tokenize_docs(docs, lang)
        if "ner" in tasks:
            res["ne"] = [list(filter(None, [remove_stops(e) for e in d.ents])) for d in tokenized_docs]
        if "noun_chunks" in tasks:
            res["noun_chunks"] = [list(filter(None, [remove_stops(nc) for nc in d.noun_chunks])) for d in tokenized_docs]
        return res

    def run(self):
        while True:
            k, docs, lang, tasks = self.next_item()
            if isinstance(docs, str):
                docs = [docs]
            try:
                res = {}
                res["id"] = k
                if tasks[0] == "sent_tokenize":
                    res["sents"] = self.sent_tokenize(docs, lang)
                else:
                    res.update(self.nlp(docs, lang, tasks))
                self.redis_store.set(k, json.dumps(res))
            except Exception as e:
                self.logger.error(e)
                self.logger.error(docs)


def remove_stops(phrase):
    while len(phrase) > 0 and phrase[0].is_stop:
        phrase = phrase[1:]
    while len(phrase) > 0 and phrase[-1].is_stop:
        phrase = phrase[:-1]
    try:
        res = phrase.text
        if res:
            return res
        else:
            return None
    except Exception:
        return None