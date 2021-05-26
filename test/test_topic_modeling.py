#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
sys.path.append(config_path)
from connlp.analysis import TopicModel


## LDA Topic Modeling
def test_topic_modeing():
    global docs, num_topics

    lda_model = TopicModel(docs=docs, num_topics=num_topics)

    # Model Training
    parameters = {
        'iterations': 100,
        'alpha': 0.7,
        'eta': 0.05,
    }
    lda_model.learn(parameters=parameters)
    print(type(lda_model.model))
    print('Coherence: {}'.format(lda_model.coherence))

    # Topic assignment
    lda_model.assign()
    print(lda_model.tag2topic)
    print(lda_model.topic2tag)
    # show_assigned_docs(lda_model)

def show_assigned_docs(lda_model):
    for topic_id in sorted(lda_model.topic2tag):
        assigned_tags = lda_model.topic2tag[topic_id]
        print('\n<<Topic #{}>>'.format(topic_id))
        for tag in assigned_tags:
            print('  | {}: {}'.format(tag, ' '.join(docs[tag])))


## Run
if __name__ == '__main__':
    num_topics = 2
    docs = {'doc1': ['I', 'am', 'a', 'boy'],
            'doc2': ['He', 'is', 'a', 'boy'],
            'doc3': ['Cat', 'on', 'the', 'table'],
            'doc4': ['Mike', 'is', 'a', 'boy'],
            'doc5': ['Dog', 'on', 'the', 'table'],
            }

    test_topic_modeing()