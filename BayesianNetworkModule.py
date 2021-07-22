# -*- coding: UTF-8 -*-

import pickle
import csv
import networkx as nx
from ConceptNetModule import ConceptNetModule


class BayesianNetworkModule:
    def __init__(self):
        self.observed_nodes = set()
        self.conceptnet_digraph = nx.DiGraph()
        self.bayesian_network = dict()
        self.type_graph = None

    def initialize(self, type_graph, observed_nodes, info_graph):
        # Load the queried concept graph file
        # with open('./LocalTempData/queriedConceptGraph_{}.pickle'.format(type_graph), 'rb') as file:
        # 	self.conceptnet_digraph = pickle.load(file)

        self.observed_nodes = observed_nodes
        self.conceptnet_digraph = info_graph
        self.bayesian_network = dict()
        self.type_graph = type_graph

    def build_bayesian_network(self):
        # Make sure self.bayesian_network is directed acyclic graph
        # Generate a bayesian network based on the conceptnet digraph (for Gibbs Sampling)
        if nx.is_directed_acyclic_graph(self.conceptnet_digraph):
            # print(self.type_graph, ' : ', self.conceptnet_digraph.nodes())
            # Record the match table for node and index (dictionary)
            node2code = dict()
            code2node = dict()

            # Save the bayesian network data in csv (for C++)
            bayesian_network_csvdata = list()

            for code, node in enumerate(self.conceptnet_digraph):
                node2code[node] = code
                code2node[code] = node

            # print(node2code)

            # Create the bayesian for each node
            for concept_node in self.conceptnet_digraph.nodes():
                concept_code = node2code[concept_node]
                predecessors = list(self.conceptnet_digraph.predecessors(concept_node))
                # If it does not have parent nodes, the parent nodes and weight will be set -1
                bayesian_network_csvdata.append([concept_code])  # For current node
                if len(predecessors) == 0:  # This concept node is root
                    bayesian_network_csvdata.append([-1])  # For parent nodes
                    bayesian_network_csvdata.append(
                        [-1]
                    )  # For weight for each parent nodes
                else:
                    bayesian_network_csvdata.append(
                        [node2code[parent] for parent in predecessors]
                    )
                    bayesian_network_csvdata.append(
                        [
                            self.conceptnet_digraph[parent][concept_node]["weight"]
                            for parent in predecessors
                        ]
                    )

            # Save code to node table
            # with open('./LocalTempData/code2node_{}.pickle'.format(self.type_graph), 'wb') as file:
            # 	pickle.dump(code2node, file)

            # Save the observed node with the type of code and bayesin network for c++
            self.save_observed_nodes(node2code)
            self.save_bayesian_network(bayesian_network_csvdata)

            return code2node

    def save_observed_nodes(self, dict_node2code):
        # with open('./LocalTempData/observedNodes_{}.pickle'.format(self.type_graph), 'rb') as file:
        # 	observed_nodes = pickle.load(file)
        observed_nodes_csv = [dict_node2code[node] for node in self.observed_nodes]

        with open(
            "./LocalTempData/codeObservedNodes_{}.csv".format(self.type_graph), "w"
        ) as file:
            writer = csv.writer(file)
            writer.writerows([observed_nodes_csv])

    def save_bayesian_network(self, csvdata):
        with open(
            "./LocalTempData/bayesianNetwork_{}.csv".format(self.type_graph), "w"
        ) as file:
            writer = csv.writer(file)
            writer.writerows(csvdata)


def main():
    print("BayesianNetworkModule")


if __name__ == "__main__":
    main()
