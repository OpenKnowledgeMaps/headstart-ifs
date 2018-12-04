from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model


class Documents(Model):
    __keyspace__ = 'documents'

    id = columns.UUID(primary_key=True)
    doi = columns.Text(index=True)
    base = columns.Text(index=True)
    core = columns.Text(index=True)
    pubmed = columns.Text(index=True)
    linkedcat = columns.Text(index=True)
    nouns = columns.List(columns.Text)
    noun_chunks = columns.List(columns.Text)
    entities = columns.List(columns.Text)
    topics = columns.List(columns.Text)
    hypernyms = columns.List(columns.Text)
    meshterms = columns.List(columns.Text)
