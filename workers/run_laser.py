import os
import json
import redis
from laser.src.laser import LaserEmbedder


if __name__ == '__main__':
    with open("redis_config.json") as infile:
        redis_config = json.load(infile)

    redis_store = redis.StrictRedis(**redis_config)
    le = LaserEmbedder(redis_store=redis_store,
                       loglevel=os.environ.get("LASER_LOGLEVEL", "INFO"))
    le.run()
