import os
import json
import redis
from nlp.src.nlp import SpacySentencizer


if __name__ == '__main__':
    with open("redis_config.json") as infile:
        redis_config = json.load(infile)

    redis_store = redis.StrictRedis(**redis_config)
    nlp = SpacySentencizer(redis_store=redis_store,
                           loglevel=os.environ.get("LASER_LOGLEVEL", "INFO"))
    nlp.run()