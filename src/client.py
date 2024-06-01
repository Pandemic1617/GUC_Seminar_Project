import time
import json
import numpy as np
from client_env import ATTACK_TYPE,CLIENT_ID
from consts import PUBSUB_COORDINATOR_RECEIVE_KEY,PUBSUB_COORDINATOR_SEND_KEY,LABELS
import random
from myredis import r
from methods import create_model,read_file



def deterministic_random(seed):
    return random.Random(seed).random()

def parse_model_update(data):
    x = np.array(data['x'])
    y = np.array(data['y'])
    return x,y

def parse_query(data):
    return np.array(data['query'])



x_local, y_local = read_file('../data/client'+CLIENT_ID+'.csv')
x_global, y_global = np.array([]), np.array([])

p = r.pubsub()
p.subscribe(PUBSUB_COORDINATOR_SEND_KEY)


print("registering")
while True:
    r.publish(PUBSUB_COORDINATOR_RECEIVE_KEY, json.dumps({'type':'register','client_id':CLIENT_ID}))
    time.sleep(0.1)
    message = p.get_message( ignore_subscribe_messages=True)
    if message:
        data = json.loads(message['data'])
        if data['type'] == 'registered':
            break


for message in p.listen():
    if message['type'] != 'message': continue
    data = json.loads(message['data'])
    if data['type'] == "update":
        x,y = parse_model_update(data)
        x_global = x
        y_global = y
        print("updated model")
    elif data['type'] == "query":
        query = parse_query(data)
        model = create_model(
            np.concatenate([x_local, x_global],axis=0),
            np.concatenate([y_local, y_global],axis=0))
        result = model.predict(query)

        if ATTACK_TYPE == "none":
            pass
        elif ATTACK_TYPE == "random_mislabel":
            result = np.array([random.choice(LABELS) for _ in range(len(result))])
        elif ATTACK_TYPE == "smart_mislabel":
            choices = np.array([ int(deterministic_random(np.sum(query[i]))* (len(LABELS) -1)) for i in range(len(result))])
            choices = np.array([choice +(1 if choice>= LABELS.index(result[i]) else 0) for (i, choice) in enumerate(choices)])
            result = np.array([LABELS[choice] for choice in choices])

        r.publish(PUBSUB_COORDINATOR_RECEIVE_KEY, json.dumps({'type':'result','client_id':CLIENT_ID,'result':result.tolist()}))
        print("responded to query")
    elif data['type'] == "registered": continue
    else:
        assert False








