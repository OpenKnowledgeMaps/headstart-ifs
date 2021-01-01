#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import struct
import tempfile
from pathlib import Path
import numpy as np
import msgpack
import msgpack_numpy as mnp
from LASER.source.lib.text_processing import Token, BPEfastApply
from LASER.source.embed import *


formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
mnp.patch()

class GUSEMEmbedder(object):

    def __init__(self, redis_store, loglevel, model_dir):
        self.redis_store = redis_store
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(loglevel)
        self.logger.addHandler(handler)
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        # encoder
        encoder_path = os.path.join(self.model_dir, "bilstm.93langs.2018-12-26.pt")
        self.bpe_codes_path = os.path.join(self.model_dir, "93langs.fcodes")
        self.logger.info('Encoder: loading %s' % encoder_path)
        self.encoder = SentenceEncoder(encoder_path,
                                       max_sentences=None,
                                       max_tokens=12000,
                                       sort_kind='mergesort',
                                       cpu=False)
        self.logger.info("Using CUDA: %s" % str(self.encoder.use_cuda))

    def next_item(self):
        queue, msg = self.redis_store.blpop("laser")
        msg = json.loads(msg)
        k = msg.get('id')
        sents = msg.get('sents')
        return k, sents

    def vectorize(self, sents):

    def run(self):
        while True:
            k, sents = self.next_item()
            try:
                embeddings = self.vectorize(sents)
                self.redis_store.set(k, mnp.packb(embeddings))
            except Exception as e:
                self.logger.error(e)