from flask import Flask

# from app.models import Documents
from databases import db


def new_ifs_app(settings_override=None):
    from blueprints.ifs.views import app as ifs
    app = Flask('ifs', instance_relative_config=True)

    app.config.from_object('config.settings')
    app.config.from_pyfile('settings.py', silent=True)
    # csrf = CSRFProtect(app)
    app.config.update(dict(
        CASSANDRA_KEYSPACE="documents"
    ))

    db.init_app(app)

    app.register_blueprint(ifs)

    return app


def new_tagging_app(settings_override=None):
    from blueprints.tagger.views import app as tagger
    app = Flask('tagger', instance_relative_config=True)

    app.config.from_object('config.settings')
    app.config.from_pyfile('settings.py', silent=True)
    # csrf = CSRFProtect(app)
    app.config.update(dict(
        CASSANDRA_KEYSPACE="documents"
    ))
    # db.init_app(app)
    # db.create_keyspace_simple('documents', replication_factor=1)
    # db.sync_table(Documents, keyspaces=['documents'], connections=None)

    app.register_blueprint(tagger)

    return app


def new_summarization_app(settings_override=None):
    from blueprints.summarizer.views import app as summarizer
    app = Flask('summarizer', instance_relative_config=True)

    app.config.from_object('config.settings')
    app.config.from_pyfile('settings.py', silent=True)
    # csrf = CSRFProtect(app)
    app.config.update(dict(
        CASSANDRA_KEYSPACE="documents"
    ))

    app.register_blueprint(summarizer)

    return app


if __name__ == "__main__":
    app = new_tagging_app()
    app.run(host='0.0.0.0', port=5002)
