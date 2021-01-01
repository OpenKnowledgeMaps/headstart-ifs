#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import tensorflow as tf
import msgpack
import msgpack_numpy as mnp


formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
mnp.patch()

class GUSEMEmbedder(object):

    def __init__(self, redis_store, model_dir, loglevel):
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
        self.model = hub.load(self.model_dir)
        self.logger.info("Using CUDA: %s" % str(tf.test.is_gpu_available()))

    def next_item(self):
        queue, msg = self.redis_store.blpop("gusem")
        msg = json.loads(msg)
        k = msg.get('id')
        sents = msg.get('sents')
        return k, sents

    def vectorize(self, sents):
        embeddings = self.model(sents)
        return embeddings.numpy()

    def run(self):
        while True:
            k, sents = self.next_item()
            try:
                embeddings = self.vectorize(sents)
                self.redis_store.set(k, mnp.packb(embeddings))
            except Exception as e:
                self.logger.error(e)