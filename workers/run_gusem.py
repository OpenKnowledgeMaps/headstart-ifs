import os
import json
import redis
from gusem.src.gusem import GUSEMEmbedder


if __name__ == '__main__':
    with open("redis_config.json") as infile:
        redis_config = json.load(infile)

    redis_store = redis.StrictRedis(**redis_config)
    ge = GUSEMEmbedder(redis_store=redis_store,
                       model_dir="/models/universal-sentence-encoder-multilingual_3",
                       loglevel=os.environ.get("GUSEM_LOGLEVEL", "INFO"))
    ge.run()
