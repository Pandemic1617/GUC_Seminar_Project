import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from coordinator_env import CLIENT_COUNT,DEFENSE_TYPE
from consts import PUBSUB_COORDINATOR_RECEIVE_KEY,PUBSUB_COORDINATOR_SEND_KEY,IMAGES_PER_ROUND,K,NUM_ROUNDS,METRIC
import random
from myredis import r
from methods import accuracy_score,create_model,evaluate_model,read_file


x_train, y_train = read_file('../data/coordinator_model.csv')
x_test, y_test = read_file('../data/coordinator_test.csv')
x_query, _ = read_file('../data/coordinator_query.csv')

results_list = []


clients = []

p = r.pubsub()
p.subscribe(PUBSUB_COORDINATOR_RECEIVE_KEY)
print("waiting for clients to register")

# p.get_message()
for message in p.listen():
    if message['type'] != 'message':
        continue
    data = json.loads(message['data'])
    if data['type'] == 'register':
        id = data['client_id']
        r.publish(PUBSUB_COORDINATOR_SEND_KEY, json.dumps({'type':'registered','client_id':id}))
        if id in clients:
            continue
        clients.append(data['client_id'])
        if len(clients) == CLIENT_COUNT:
            break

clients = sorted(clients)
print("all clients registered", clients)

def send_query_to_clients(query):
    r.publish(PUBSUB_COORDINATOR_SEND_KEY, json.dumps({'type':'query','query':query}))

def get_all_results_from_clients(query):
    p = r.pubsub()
    p.subscribe(PUBSUB_COORDINATOR_RECEIVE_KEY)
    results = []
    send_query_to_clients(query)
    for message in p.listen():
        if message['type'] != 'message': continue
        data = json.loads(message['data'])
        if data['type'] == 'result':
            results.append((data['client_id'], np.array(data['result'])))
        if len(results) == CLIENT_COUNT:
            break
    sorted_results = sorted(results, key=lambda x: x[0])
    return np.array([result[1] for result in sorted_results])
    
def update_client_model(x,y):
    r.publish(PUBSUB_COORDINATOR_SEND_KEY, json.dumps({'type':'update','x':x.tolist(),'y':y.tolist()}))


def get_mode(results):
    freq = {}
    for result in results:
        if result not in freq:
            freq[result] = 1
        else:
            freq[result] += 1
    return max(freq.keys(), key=freq.get)


print("initial metrics: ", evaluate_model(create_model(x_train, y_train), x_test, y_test))


for round_num in range(NUM_ROUNDS):
    update_client_model(x_train, y_train)


    start_ind = round_num * IMAGES_PER_ROUND
    end_ind = start_ind + IMAGES_PER_ROUND
    query = x_query[start_ind:end_ind]

    results = get_all_results_from_clients(query.tolist())
    if DEFENSE_TYPE == 'none':
        results = random.choice(results)
    elif DEFENSE_TYPE == 'mode':
        results = np.array(list(map(get_mode, results.T)))
    else:
        assert False
    assert len(query) == len(results), f"query and results length mismatch: {len(query)} != {len(results)}"
    x_train = np.concatenate((x_train, query),axis=0)
    y_train = np.concatenate((y_train, results),axis=0)


    round_evaluation = evaluate_model(create_model(x_train, y_train), x_test, y_test)
    print("round: ", round_num, "metrics: ", round_evaluation)
    round_evaluation['round_num'] = round_num
    results_list.append(round_evaluation)

results_df = pd.DataFrame(results_list)
results_df.to_csv('../results/results.csv', index=False)
print("saved results to file")
