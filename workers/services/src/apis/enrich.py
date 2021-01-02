import time
import uuid
import json
import struct
import numpy as np
import msgpack
import msgpack_numpy as mnp
from flask import Blueprint, redirect, request, jsonify, make_response
from flask_restx import Namespace, Resource, fields
import redis


mnp.patch()

with open("redis_config.json") as infile:
    redis_config = json.load(infile)

redis_store = redis.StrictRedis(**redis_config)

enrich_ns = Namespace("enrich", description="OKMAps item enrichment operations")


def lang_detect(docs, batch=False):
    if not batch:
        docs = [docs]
    k = str(uuid.uuid4())
    d = {"id": k, "docs": docs}
    redis_store.rpush("batch_lang_detect", json.dumps(d))
    while True:
        result = redis_store.get(k)
        if result is not None:
            result = json.loads(result.decode('utf-8'))
            langs = result.get('langs')
            redis_store.delete(k)
            break
        time.sleep(0.5)
    if not batch:
        return langs[0]
    else:
        return langs


@enrich_ns.route('/lang_detect')
class LangDetect(Resource):
    def post(self):
        enrich_ns.logger.debug("lang_detect")
        try:
            result = {"success": False}
            r = request.get_json()
            doc = r.get('doc')
            assert isinstance(doc, str)
            if len(doc) is 0:
                result["msg"] = "No document provided"
                return jsonify(result, 400)
            result["detected_langs"] = lang_detect(doc, False)
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            enrich_ns.logger.error(e)
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)

@enrich_ns.route('/lang_detect/batch')
class LangDetectBatch(Resource):
    def post(self):
        enrich_ns.logger.debug("lang_detect/batch")
        try:
            result = {"success": False}
            r = request.get_json()
            docs = r.get('docs')
            assert isinstance(docs, list)
            if len(docs) is 0:
                result["msg"] = "No documents provided"
                return jsonify(result, 400)
            result["detected_langs"] = lang_detect(docs, True)
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            enrich_ns.logger.error(e)
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)


def sent_tokenize(docs, lang):
    k = str(uuid.uuid4())
    d = {"id": k, "docs": docs, "lang": lang}
    redis_store.rpush("sent_tokenize", json.dumps(d))
    while True:
        result = redis_store.get(k)
        if result is not None:
            result = json.loads(result.decode('utf-8'))
            sents = result.get('sents')
            redis_store.delete(k)
            break
        time.sleep(0.5)
    return sents


@enrich_ns.route('/sent_tokenize')
class SentTokenize(Resource):
    def post(self):
        enrich_ns.logger.debug("sent_tokenize")
        try:
            result = {"success": False}
            r = request.get_json()
            docs = r.get('docs')
            lang = r.get('lang')
            result["sents"] = sent_tokenize(docs, lang)
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                    200,
                                    headers)
        except Exception as e:
            enrich_ns.logger.error(e)
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                    500,
                                    headers)


def sent_embed_gusem(sents):
    k = str(uuid.uuid4())
    d = {"id": k, "sents": sents}
    redis_store.rpush("gusem", json.dumps(d))
    while True:
        embeddings = redis_store.get(k)
        if embeddings is not None:
            redis_store.delete(k)
            break
        time.sleep(0.1)
    return embeddings


@enrich_ns.route('/sent_embed/gusem')
class SentEmbed(Resource):
    def post(self):
        enrich_ns.logger.debug("sent_embed/gusem")
        try:
            result = {"success": False}
            r = request.get_json()
            sents = r.get('sents')
            embeddings = sent_embed_gusem(sents)
            headers = {'ContentType': 'application/x-msgpack'}
            return make_response(embeddings,
                                    200,
                                    headers)
        except Exception as e:
            enrich_ns.logger.error(e)
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                    500,
                                    headers)

# @enrich_ns.route('/tag')
# def tag():
#     response = {"success": False}
#     if request.method == 'POST':
#         r = request.get_json()
#         lang = r.get('lang')
#         doc = json.loads(r.get('doc', "[]"))[0]
#         if lang is None:
#             response["msg"] = "No language provided, add <lang> to request"
#             return jsonify(response)

#         k = str(uuid.uuid4())
#         d = {"id": k, "doc": doc}
#         redis_store.rpush("tag_"+lang, json.dumps(d))
#         while True:
#             result = redis_store.get(k)
#             if result is not None:
#                 result = json.loads(result.decode('utf-8'))
#                 noun_chunks = result.get('noun_chunks')
#                 response["noun_chunks"] = noun_chunks
#                 redis_store.delete(k)
#                 break
#             time.sleep(0.5)
#         response["success"] = True
#     return jsonify(response)


# @enrich_ns.route('/batch_pos')
# def batch_pos():
#     response = {"success": False}
#     if request.method == 'POST':
#         r = request.get_json()
#         lang = r.get('lang')
#         docs = r.get('docs', [])
#         if lang is None:
#             response["msg"] = "No language provided, add <lang> to request"
#             return jsonify(response)
#         if len(docs) is 0:
#             response["msg"] = "No documents provided"
#             return jsonify(response)

#         k = str(uuid.uuid4())
#         d = {"id": k, "docs": docs}
#         redis_store.rpush("batch_pos_"+lang, json.dumps(d))
#         while True:
#             result = redis_store.get(k)
#             if result is not None:
#                 result = json.loads(result.decode('utf-8'))
#                 noun_chunks = result.get('noun_chunks')
#                 response["noun_chunks"] = noun_chunks
#                 redis_store.delete(k)
#                 break
#             time.sleep(0.5)
#         response["success"] = True
#     return jsonify(response)


# @enrich_ns.route('/batch_ner')
# def batch_ne():
#     response = {"success": False}
#     if request.method == 'POST':
#         r = request.get_json()
#         lang = r.get('lang')
#         docs = r.get('docs', [])
#         if lang is None:
#             response["msg"] = "No language provided, add <lang> to request"
#             return jsonify(response)
#         if len(docs) is 0:
#             response["msg"] = "No documents provided"
#             return jsonify(response)

#         k = str(uuid.uuid4())
#         d = {"id": k, "docs": docs}
#         redis_store.rpush("batch_ner_"+lang, json.dumps(d))
#         while True:
#             result = redis_store.get(k)
#             if result is not None:
#                 result = json.loads(result.decode('utf-8'))
#                 entities = result.get('entities')
#                 response["entities"] = entities
#                 redis_store.delete(k)
#                 break
#             time.sleep(0.5)
#         response["success"] = True
#     return jsonify(response)


# @enrich_ns.route('/batch_tokenize')
# def batch_tokenize():
#     response = {"success": False}
#     if request.method == 'POST':
#         r = request.get_json()
#         lang = r.get('lang')
#         docs = r.get('docs', [])
#         if lang is None:
#             response["msg"] = "No language provided, add <lang> to request"
#             return jsonify(response)
#         if len(docs) is 0:
#             response["msg"] = "No documents provided"
#             return jsonify(response)

#         k = str(uuid.uuid4())
#         d = {"id": k, "docs": docs}
#         redis_store.rpush("batch_tokenize_"+lang, json.dumps(d))
#         while True:
#             result = redis_store.get(k)
#             if result is not None:
#                 result = json.loads(result.decode('utf-8'))
#                 tokens = result.get('tokens')
#                 response['tokens'] = tokens
#                 redis_store.delete(k)
#                 break
#             time.sleep(0.5)
#         response["success"] = True
#     return jsonify(response)
