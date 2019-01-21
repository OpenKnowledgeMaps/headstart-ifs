import time
import uuid
import json
from flask import Blueprint, redirect, request, jsonify
import redis


redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)

app = Blueprint('tagger', __name__)


@app.route('/')
def main():
    return "Hello World!"


@app.route('/batch_lang_detect', methods=['GET', 'POST'])
def batch_lang_detect():
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        docs = r.get('docs', [])
        if len(docs) is 0:
            response["msg"] = "No documents provided"
            return jsonify(response)

        k = str(uuid.uuid4())
        d = {"id": k, "docs": docs}
        redis_store.rpush("batch_lang_detect", json.dumps(d))
        while True:
            result = redis_store.get(k)
            if result is not None:
                result = json.loads(result.decode('utf-8'))
                langs = result.get('detected_langs')
                response["detected_langs"] = langs
                redis_store.delete(k)
                break
            time.sleep(0.5)
        response["success"] = True
    return jsonify(response)


@app.route('/tag', methods=['GET', 'POST'])
def tag():
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        lang = r.get('lang')
        doc = json.loads(r.get('doc', "[]"))[0]
        if lang is None:
            response["msg"] = "No language provided, add <lang> to request"
            return jsonify(response)

        k = str(uuid.uuid4())
        d = {"id": k, "doc": doc}
        redis_store.rpush("tag_"+lang, json.dumps(d))
        while True:
            result = redis_store.get(k)
            if result is not None:
                result = json.loads(result.decode('utf-8'))
                noun_chunks = result.get('noun_chunks')
                response["noun_chunks"] = noun_chunks
                redis_store.delete(k)
                break
            time.sleep(0.5)
        response["success"] = True
    return jsonify(response)


@app.route('/batch_pos', methods=['GET', 'POST'])
def batch_pos():
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        lang = r.get('lang')
        docs = r.get('docs', [])
        if lang is None:
            response["msg"] = "No language provided, add <lang> to request"
            return jsonify(response)
        if len(docs) is 0:
            response["msg"] = "No documents provided"
            return jsonify(response)

        k = str(uuid.uuid4())
        d = {"id": k, "docs": docs}
        redis_store.rpush("batch_pos_"+lang, json.dumps(d))
        while True:
            result = redis_store.get(k)
            if result is not None:
                result = json.loads(result.decode('utf-8'))
                noun_chunks = result.get('noun_chunks')
                response["noun_chunks"] = noun_chunks
                redis_store.delete(k)
                break
            time.sleep(0.5)
        response["success"] = True
    return jsonify(response)


@app.route('/batch_ner', methods=['GET', 'POST'])
def batch_ne():
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        lang = r.get('lang')
        docs = r.get('docs', [])
        if lang is None:
            response["msg"] = "No language provided, add <lang> to request"
            return jsonify(response)
        if len(docs) is 0:
            response["msg"] = "No documents provided"
            return jsonify(response)

        k = str(uuid.uuid4())
        d = {"id": k, "docs": docs}
        redis_store.rpush("batch_ner_"+lang, json.dumps(d))
        while True:
            result = redis_store.get(k)
            if result is not None:
                result = json.loads(result.decode('utf-8'))
                entities = result.get('entities')
                response["entities"] = entities
                redis_store.delete(k)
                break
            time.sleep(0.5)
        response["success"] = True
    return jsonify(response)


@app.route('/batch_tokenize', methods=['GET', 'POST'])
def batch_tokenize():
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        lang = r.get('lang')
        docs = r.get('docs', [])
        if lang is None:
            response["msg"] = "No language provided, add <lang> to request"
            return jsonify(response)
        if len(docs) is 0:
            response["msg"] = "No documents provided"
            return jsonify(response)

        k = str(uuid.uuid4())
        d = {"id": k, "docs": docs}
        redis_store.rpush("batch_tokenize_"+lang, json.dumps(d))
        while True:
            result = redis_store.get(k)
            if result is not None:
                result = json.loads(result.decode('utf-8'))
                tokens = result.get('tokens')
                response['tokens'] = tokens
                redis_store.delete(k)
                break
            time.sleep(0.5)
        response["success"] = True
    return jsonify(response)
