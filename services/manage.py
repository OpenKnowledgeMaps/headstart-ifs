from cassandra.cluster import Cluster
from cassandra.cqlengine import connection
from cassandra.cqlengine.management import sync_table

from flask.cli import FlaskGroup
from app import new_ifs_app
from models import Documents

app = new_ifs_app()
cli = FlaskGroup(create_app=new_ifs_app)


@cli.command('recreate_db')
def recreate_db():
    """Cleans everything (!) and sets up database"""

    cluster = Cluster(['192.168.1.1', '192.168.1.2', '127.0.0.1'])
    session = cluster.connect()
    connection.set_session(session)

    session.execute('DROP KEYSPACE IF EXISTS documents;')
    session.execute("""CREATE KEYSPACE IF NOT EXISTS documents
                       WITH REPLICATION = {
                        'class' : 'SimpleStrategy', 'replication_factor' : 1
                        }
                    """)
    session.set_keyspace('documents')
    session.execute('DROP TABLE IF EXISTS documents;')
    sync_table(Documents)
