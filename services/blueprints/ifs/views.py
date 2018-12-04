import uuid
import time
import json
import redis
from flask import Blueprint, redirect, jsonify, request
from models import Documents


redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)


app = Blueprint('documents', __name__)


@app.route('/')
def main():
    return redirect('/index')


@app.route('/noun_chunks/<index>/<doc_id>', methods=['GET', 'POST'])
def noun_chunks(index, doc_id):
    response = {"success": False}
    doc = (Documents.objects
                    .filter(getattr(Documents, index) == doc_id)
                    .first())
    if doc is None:
        new = {'id': uuid.uuid4(), index: doc_id}
        doc = Documents.create(**new)
    if request.method == 'GET':
        if doc is not None:
            nc = doc.noun_chunks
        response["noun_chunks"] = nc
        response["success"] = True
        return jsonify(response)
    if request.method == 'POST':
        r = request.get_json()
        nc = json.loads(r.get('noun_chunks', []))
        update = {'noun_chunks': nc}
        doc.update(**update)
        response["success"] = True
        return jsonify(response)


@app.route('/entities/<index>/<doc_id>', methods=['GET', 'POST'])
def entities(index, doc_id):
    response = {"success": False}
    doc = (Documents.objects
                    .filter(getattr(Documents, index) == doc_id)
                    .first())
    if doc is None:
        new = {'id': uuid.uuid4(), index: doc_id}
        doc = Documents.create(**new)
    if request.method == 'GET':
        if doc is not None:
            ne = doc.entities
        response["entities"] = ne
        response["success"] = True
        return jsonify(response)
    if request.method == 'POST':
        r = request.get_json()
        ne = json.loads(r.get('entities', []))
        update = {'entities': ne}
        doc.update(**update)
        response["success"] = True
        return jsonify(response)


@app.route('/get_hypernyms', methods=['GET', 'POST'])
def get_or_create_hypernyms():
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        external_id = r.get('external_id')
        doc_id = r.get('doc_id')
        hypernyms = (Documents.objects
                              .filter({external_id: doc_id})
                              .first()
                              .hypernyms)
        response["hypernyms"] = hypernyms
        response["success"] = True
        return jsonify(response)


@app.route('/tag_nouns', methods=['GET', 'POST'])
def tag_nouns():
    response = {"success": False}

    if request.method == 'POST':
        r = request.get_json()
        external_id = r.get('external_id')
        doc_id = r.get('doc_id')
        lang = r.get('lang')
        text = r.get('text')

        k = uuid.uuid4()
        d = {"id": k, "text": text}
        redis_store.rpush("tag_"+lang, json.dumps(d))
        while True:
            result = json.loads(redis_store.get(k))
            if result is not None:
                nouns = result.get('nouns')
                response["nouns"] = nouns
                redis_store.delete(k)
                break
            time.sleep(1)
        doc = (Documents.objects
                        .filter({external_id: doc_id})
                        .first())
        update = {"nouns": nouns}
        doc.update(**update)
        response["success"] = True
    return jsonify(response)


@app.route('/create_hypernyms', methods=['GET', 'POST'])
def create_hypernyms():
    response = {"success": False}

    if request.method == 'POST':
        r = request.get_json()
        id_type = r.get('id_type')
        doc_id = r.get('doc_id')
        nouns = (Documents.objects
                          .filter({id_type: doc_id})
                          .first()
                          .nouns)

        k = uuid.uuid4()
        d = {"id": k, "nouns": nouns}
        redis_store.rpush("hypernymer", json.dumps(d))
        while True:
            result = json.loads(redis_store.get(k))
            if result is not None:
                hypernyms = result.get('hypernyms')
                response["hypernyms"] = hypernyms
                redis_store.delete(k)
                break
            time.sleep(1)
        doc = (Documents.objects
                        .filter({id_type: doc_id})
                        .first())
        update = {"hypernyms": hypernyms}
        doc.update(**update)
        response["success"] = True
    return jsonify(response)
