import os
import json
import redis
from lang_detect.src.lang_detect import SpacyLangDetector


if __name__ == '__main__':
    with open("redis_config.json") as infile:
        redis_config = json.load(infile)

    redis_store = redis.StrictRedis(**redis_config)
    sld = SpacyLangDetector(redis_store=redis_store,
                            loglevel=os.environ.get("LASER_LOGLEVEL", "INFO"))
    sld.run()