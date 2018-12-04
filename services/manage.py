from cassandra.cluster import Cluster
from cassandra.cqlengine import connection
from cassandra.cqlengine.management import create_keyspace_simple, drop_keyspace, drop_table, sync_table

from flask.cli import FlaskGroup
from databases import db
from app import new_ifs_app
from models import Documents

app = new_ifs_app()
cli = FlaskGroup(create_app=new_ifs_app)


@cli.command('recreate_db')
def recreate_db():
    """Cleans everything (!) and sets up database"""

    drop_keyspace('documents')
    drop_table(Documents)
    create_keyspace_simple('documents', replication_factor=1)
    sync_table(Documents)

if __name__ == '__main__':
    cli()
