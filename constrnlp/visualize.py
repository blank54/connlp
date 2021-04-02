#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import itertools
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt


class CombCounter:
    '''
    A class used to count word combinations from a list of tokenized sentences.

    Attributes
    ----------
    docs : list
        | A list of tokenized words.
    normalize : Boolean
        | if True, the counter normalizes the weights based on the distance between words within the sentence.
        | elif False, it counts the frequency.
    
    Methods
    -------
    simple_dict
        | Counts word combinations from docs.
        | Returns a dictionary of which the key is 'WORD[A]__WORD[B]' and the value is occurred frequency.
    '''

    def __init__(self, docs, **kwargs):
        self.docs = docs
        self.normalize = kwargs.get('normalize', False)

    def simple_dict(self):
        count_dict = defaultdict(int)
        for sent in self.docs:
            for pair in itertools.combinations(sent, 2):
                count_dict['__'.join(pair)] += 1

        return count_dict

    ## TODO: counting method with normalized weights based on the distance between words within the sentence.


class Visualizer:
    '''
    A class used to visualize text mining results.

    Methods
    -------
    network(docs=None, show=False, fig_width=10, fig_height=8, dpi=300, arrows=False, node_size=10, font_size=8, width=0.6, edge_color='grey', node_color='purple', with_labels=False, facecolor='white', alpha=0, edgecolor='white'))
        | Returns a word network.
    '''

    def network(self, docs, show=False, fig_width=10, fig_height=8, dpi=300, arrows=False, node_size=10, font_size=8, width=0.6, edge_color='grey', node_color='purple', with_labels=False, facecolor='white', alpha=0, edgecolor='white'):
        '''
        A method to return word network

        Copyright (C) 2004-2021, NetworkX Developers
        Aric Hagberg <hagberg@lanl.gov>
        Dan Schult <dschult@colgate.edu>
        Pieter Swart <swart@lanl.gov>
        All rights reserved.

        Attributes
        ----------
        docs : list
            | A list of tokenized words.
        show : Boolean
            | Whether show the figure or not (default : False).
        
        NOTE: other attributes follow the original documentation of NetworkX.
        '''

        data = CombCounter(docs=docs)
        graph = nx.DiGraph()
        for key, score in data.simple_dict().items():
            node_from, node_to = key.split('__')
            graph.add_edge(node_from, node_to, weight=score)
        pos = nx.spring_layout(graph, k=0.08)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        nx.draw_networkx(
            graph, pos,
            arrows=arrows,
            node_size=node_size,
            font_size=font_size,
            width=width,
            edge_color=edge_color,
            node_color=node_color,
            with_labels=with_labels,
            ax=ax)

        for key, value in pos.items():
            ax.text(x=value[0],
                    y=value[1]+0.025,
                    s=key,
                    bbox=dict(facecolor=facecolor, alpha=alpha, edgecolor=edgecolor),
                    horizontalalignment='center',
                    fontsize=font_size)

        if show:
            plt.show()