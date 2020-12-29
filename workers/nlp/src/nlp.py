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
                              
class SpacySentencizer(object):

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
        self.pipes = {}
        for cl, source in core_models.items():
            self.logger.debug("Loading model: %s" % cl)
            self.pipes[cl] = spacy.load("%s_core_%s_sm" % (cl, source),
                                        disable=["ner", "entity_linker",
                                                 "textcat", "entity_ruler",
                                                 "merge_noun_chunks",
                                                 "merge_entities",
                                                 "merge_subtokens"])
        self.logger.debug("Models loaded")

    def next_item(self):
        queue, msg = self.redis_store.blpop("sent_tokenize")
        msg = json.loads(msg)
        k = msg.get('id')
        docs = msg.get('docs')
        lang = msg.get('lang')
        return k, docs, lang

    def sent_tokenize(self, docs, lang):
        assert isinstance(docs, list)
        self.logger.debug(docs)
        for d in docs:
            assert isinstance(d, str), "assert failed: isinstance(doc, str)"
        if lang in core_models:
            nlp = self.pipes.get(lang)
        else:
            nlp = self.pipes.get("en")
        tokenized_docs = [d for d in nlp.pipe(docs, batch_size=100, n_threads=5)]
        self.logger.debug(tokenized_docs)
        sents = [[s.text for s in d.sents] for d in tokenized_docs]
        self.logger.debug(sents)
        return sents

    def run(self):
        while True:
            k, docs, lang = self.next_item()
            try:
                res = {}
                res["id"] = k
                res["sents"] = self.sent_tokenize(docs, lang)
                self.redis_store.set(k, json.dumps(res))
            except Exception as e:
                self.logger.error(e)