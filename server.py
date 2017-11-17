from flask import Flask, request, jsonify
from predict_client.prod_client import PredictClient
app = Flask('Aura-Server')
import numpy as np
import tensorflow as tf
import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

@app.route("/predict", methods=['POST'])
def handle_prediction():
    sentence = request.form['sentence'].lower()
    client = PredictClient('localhost:8000', 'default', 1510811069)
    sentence = tf.constant(sentence.split(' '))
    index_lookup_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file='./vocab_list.txt',
    )
    ids = index_lookup_table.lookup(sentence)
    with tf.Session().as_default():
        tf.tables_initializer().run()
        ids = ids.eval()
    ids = np.append(ids, [2]) #append stop token.
    print(ids)
    request_data = { 'input': np.array([ids]), 'input_sz': np.array([len(ids)]) }
    proto = client.predict(request_data)
    print(proto)
    if proto:
        predictions = tf.contrib.util.make_ndarray(proto)
        api_requests = []
        for pred in predictions:
           api_request = []
           for x in pred:
               api_requests.append(most_common(x).decode('utf-8'))
        print(api_requests)
        return jsonify(api_requests)
    else:
        return jsonify("Hmm, sorry I'm still learning how to speak that api, can you try it again later?")

