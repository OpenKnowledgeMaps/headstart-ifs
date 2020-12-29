import os
import sys
from flask import Flask
from flask_restx import Api
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import logging

from apis.enrich import enrich_ns
from apis.items import items_ns
# from database import db

from config import settings
from utils.monkeypatches import ReverseProxied


def api_patches(app, settings):

    api_fixed = Api(
        app,
        title="Head Start API",
        description="Head Start API demo",
        version="0.1",
        prefix='/ifs/',
        doc="/api/enrich/docs")
    if settings.BEHIND_PROXY:
        api_fixed.behind_proxy = True
    return api_fixed


app = Flask('v1', instance_relative_config=True)
app.config.from_object('config.settings')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(app.logger.level)
# db.init_app(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_port=1, x_for=1, x_host=1, x_prefix=1)
app.wsgi_app = ReverseProxied(app.wsgi_app)
CORS(app, expose_headers=["Content-Disposition", "Access-Control-Allow-Origin"])

api = api_patches(app, settings)
api.add_namespace(enrich_ns, path='/enrich')
api.add_namespace(items_ns, path='/items')
app.logger.debug(app.config)
app.logger.debug(app.url_map)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=True)