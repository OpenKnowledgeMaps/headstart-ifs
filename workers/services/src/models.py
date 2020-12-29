from elasticsearch_dsl import Document, Binary, Date, Keyword, Integer, Text, DenseVector


class Item(Document):
    __keyspace__ = 'documents'

    faiss_id = Integer()
    title = Text() 
    lang = Keyword()
    url = Keyword()
    publication_date = Date()
    # categories = List(Text)
    description = Text()
    fulltext = Text()
    embeddings_doc = DenseVector(dims=1024)
    embeddings_quants = DenseVector(dims=1024)
    embeddings_sents = DenseVector(dims=1024)
    # nouns = List(Text)
    # noun_chunks = List(Text)
    # entities = List(Text)
    # topics = List(Text)
    # hypernyms = List(Text)
    # meshterms = List(Text)
