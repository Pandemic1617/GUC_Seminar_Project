import redis
from consts import REDIS_HOST
r = redis.Redis(host=REDIS_HOST)
