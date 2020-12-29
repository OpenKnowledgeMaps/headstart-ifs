import uuid
import time
import json
from flask import Blueprint, redirect, jsonify, request, make_response
from flask_restx import Namespace, Resource, fields
from models import Item as Documents
import redis


with open("redis_config.json") as infile:
    redis_config = json.load(infile)

redis_store = redis.StrictRedis(**redis_config)

items_ns = Namespace("items", description="OKMAps item store operations")


@items_ns.route('/noun_chunks/<index>/<doc_id>')
class NounChunks(Resource):

    @items_ns.produces(["application/json"])
    def get(self, index, doc_id):
        try:
            result = {"success": False}
            doc = (Documents.objects
                            .filter(getattr(Documents, index) == doc_id)
                            .first())
            if doc is None:
                new = {'id': uuid.uuid4(), index: doc_id}
                doc = Documents.create(**new)
            if doc is not None:
                nc = doc.noun_chunks
            result["noun_chunks"] = nc
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)

    @items_ns.produces(["application/json"])
    def post(self, index, doc_id):
        try:
            result = {"success": False}
            doc = (Documents.objects
                            .filter(getattr(Documents, index) == doc_id)
                            .first())
            if doc is None:
                new = {'id': uuid.uuid4(), index: doc_id}
                doc = Documents.create(**new)
            r = request.get_json()
            nc = json.loads(r.get('noun_chunks', []))
            update = {'noun_chunks': nc}
            doc.update(**update)
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)


@items_ns.route('/entities/<index>/<doc_id>')
class Entities(Resource):

    def get(self, index, doc_id):
        try:
            result = {"success": False}
            doc = (Documents.objects
                            .filter(getattr(Documents, index) == doc_id)
                            .first())
            if doc is None:
                new = {'id': uuid.uuid4(), index: doc_id}
                doc = Documents.create(**new)
            if doc is not None:
                ne = doc.entities
            result["entities"] = ne
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)

    def post(self, index, doc_id):
        try:
            result = {"success": False}
            doc = (Documents.objects
                            .filter(getattr(Documents, index) == doc_id)
                            .first())
            if doc is None:
                new = {'id': uuid.uuid4(), index: doc_id}
                doc = Documents.create(**new)
            r = request.get_json()
            ne = json.loads(r.get('entities', []))
            update = {'entities': ne}
            doc.update(**update)
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)


@items_ns.route('/get_hypernyms')
class Hypernyms(Resource):

    def post(self):
        try:
            result = {"success": False}
            r = request.get_json()
            external_id = r.get('external_id')
            doc_id = r.get('doc_id')
            hypernyms = (Documents.objects
                                .filter({external_id: doc_id})
                                .first()
                                .hypernyms)
            result["hypernyms"] = hypernyms
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)


@items_ns.route('/tag_nouns')
class TagNouns(Resource):

    def post(self):
        try:
            result = {"success": False}

            r = request.get_json()
            external_id = r.get('external_id')
            doc_id = r.get('doc_id')
            lang = r.get('lang')
            text = r.get('text')

            k = uuid.uuid4()
            d = {"id": k, "text": text}
            redis_store.rpush("tag_"+lang, json.dumps(d))
            while True:
                res = json.loads(redis_store.get(k))
                if res is not None:
                    nouns = res.get('nouns')
                    result["nouns"] = nouns
                    redis_store.delete(k)
                    break
                time.sleep(1)
            doc = (Documents.objects
                            .filter({external_id: doc_id})
                            .first())
            update = {"nouns": nouns}
            doc.update(**update)
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)


@items_ns.route('/create_hypernyms')
class CreateHypernyms(Resource):

    def post(self):
        try:
            result = {"success": False}

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
                res = json.loads(redis_store.get(k))
                if res is not None:
                    hypernyms = result.get('hypernyms')
                    result["hypernyms"] = hypernyms
                    redis_store.delete(k)
                    break
                time.sleep(1)
            doc = (Documents.objects
                            .filter({id_type: doc_id})
                            .first())
            update = {"hypernyms": hypernyms}
            doc.update(**update)
            result["success"] = True
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            result = {'success': False, 'reason': e}
            headers = {'ContentType': 'application/json'}
            return make_response(jsonify(result),
                                 500,
                                 headers)
