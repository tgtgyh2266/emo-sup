#include <algorithm>
#include <bayonet/Bayesnet.h>
#include <bayonet/Bayesnode.h>
#include <bayonet/GibbsSampler.h>
#include <bayonet/JointProbabilityTable.h>
#include <bayonet/MarginalProbabilityTable.h>
#include <boost/algorithm/string.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace std;
using namespace boost;
using namespace bayonet;

/*
 * Class to read data from csv
 */
class CSVReader {
    string file_name;
    string delimiter;

  public:
    CSVReader(string filename, string delm = ",") {
        file_name = filename;
        delimiter = delm;
    }

    vector<vector<string>> getData() {
        ifstream file(file_name);
        vector<vector<string>> data_list;
        string line = "";

        while (getline(file, line)) {
            vector<string> data_line;
            algorithm::split(data_line, line, is_any_of(","));
            data_list.push_back(data_line);
        }

        file.close();

        return data_list;
    }
};

char *int2bin(int n, int bitWidth) {
    // determine the number of bits needed ("sizeof" returns bytes)
    int nbits = bitWidth;
    char *s = (char *)malloc(nbits + 1);  // +1 for '\0' terminator
    s[nbits] = '\0';
    // forcing evaluation as an unsigned value prevents complications
    // with negative numbers at the left-most bit
    unsigned int u = *(unsigned int *)&n;
    int i;
    unsigned int mask = 1 << (nbits - 1);  // fill in values right-to-left
    for (i = 0; i < nbits; i++, mask >>= 1)
        s[i] = ((u & mask) != 0) + '0';
    return s;
}

double probability_function(double sum_weight) {
    double alpha = 0.35;
    // double theta = 0.1;

    // return (1/(1+exp(-sum_weight*alpha)) - theta)/(1-theta);
    return (1 / (1 + exp(-sum_weight * alpha)));
}

double rounding(double num, int index) {
    bool isNegative = false;

    if (num < 0) {
        isNegative = true;
        num = -num;
    }

    if (index >= 0) {
        int multiplier;
        multiplier = pow(10, index);
        num = (int)(num * multiplier + 0.5) / (multiplier * 1.0);
    }

    if (isNegative)
        num = -num;

    return num;
}

int main(int argc, char **argv) {
    string graph_type = argv[1];
    string file_name = "./LocalTempData/bayesianNetwork_" + graph_type + ".csv";
    // Get the data from csv
    CSVReader reader(file_name);
    vector<vector<string>> data_list = reader.getData();

    vector<vector<double>> parent_list, weight_list;

    // Prepare parent node and weight
    for (int i = 0; i != data_list.size(); i++) {
        if (i % 3 == 0) {
            vector<double> temp_vector;
            for (string data : data_list[i + 1]) {
                temp_vector.push_back(stod(data));
            }
            parent_list.push_back(temp_vector);

            temp_vector.clear();
            for (string data : data_list[i + 2]) {
                temp_vector.push_back(stod(data));
            }
            weight_list.push_back(temp_vector);
        }
    }

    // Set the number of state for each node (Due to our node only True or
    // False, the number is 2)
    vector<unsigned int> node_states;
    // The size of parent_list also is the number of node
    for (int i = 0; i < parent_list.size(); i++) {
        node_states.push_back(2);
    }

    // Build the bayesian network construction; First, set the number of state
    // for each node
    Bayesnet my_net(node_states);
    // Define the number of iteration to use for the sampler
    // Linear time; 25000 -> 3.6s 100000 -> 14s
    unsigned int iterations = 10000;
    GibbsSampler my_gibbs_sampler;

    // Set the bayesian network (add edge) the node is defined the type of
    // unsigned int
    for (int i = 0; i < parent_list.size(); i++) {
        for (double parent_node : parent_list[i]) {
            if (parent_node != -1) {
                my_net.AddEdge(parent_node, i);
            }
        }
    }

    // Set the conditional tables; parent node set table {0.5,0.5} (weight = -1)
    for (int i = 0; i < parent_list.size(); i++) {
        vector<double> parent_nodes = parent_list[i];
        vector<double> parent_weight = weight_list[i];

        double check_weight = 0;
        for (double weight : parent_weight)
            check_weight += weight;

        if (check_weight == -1) {  // parent node
            my_net[i].conditionalTable.SetProbabilities({}, {0.5, 0.5});
        } else {
            int num_row_table = pow(2, parent_nodes.size());
            for (int code = 0; code < num_row_table;
                 code++) {  // 2 parents: "00", "01", "10", "11"
                char *bin_code = int2bin(code, parent_nodes.size());

                vector<unsigned int> state;
                for (int j = 0; bin_code[j] != '\0'; j++) {
                    state.push_back(bin_code[j] - '0');
                }

                // The boolean of parents are 0 and 1; the weights are 2.5
                // and 1.5 then sum_weight = 0 * 2.5 + 1 * 1.5 = 1.5
                vector<double> probabilities;
                double sum_weight = 0;
                for (int index = 0; index < parent_nodes.size(); index++) {
                    sum_weight += (state[index] * parent_weight[index] - 1);
                }

                // Probability of true for each node
                double probability_true =
                  rounding(probability_function(sum_weight), 2);
                // Set False and True probability
                probabilities.push_back(1 - probability_true);
                probabilities.push_back(probability_true);

                // Set state and its corresponding probability
                my_net[i].conditionalTable.SetProbabilities(state,
                                                            probabilities);
            }
        }
    }

    // Set up inference states (observed nodes should be True)
    file_name = "./LocalTempData/codeObservedNodes_" + graph_type + ".csv";
    CSVReader observed_reader(file_name);
    data_list = observed_reader.getData();
    vector<double> observed_list;

    for (int i = 0; i != data_list.size(); i++) {
        for (string data : data_list[i]) {
            observed_list.push_back(stod(data));
        }
    }

    // Set True on the observed nodes
    for (double node_code : observed_list)
        my_net[node_code].SetEvidence(true);

    auto sample_vector = my_gibbs_sampler.AccumulateSamples(my_net, iterations);

    // Save sample matrix (iterations * number of nodes)
    // Average all sample (Add all result / iteration) Speed up compution -> Use
    // boost matrix
    numeric::ublas::matrix<double> sample_martrix(iterations,
                                                  my_net.ReturnNumberOfNodes());
    for (int i = 0; i < iterations; i++) {
        for (int num = 0; num < my_net.ReturnNumberOfNodes(); num++) {
            sample_martrix(i, num) = sample_vector[i][num];
        }
    }
    numeric::ublas::scalar_matrix<double> ones_martirx(1, iterations, 1);
    auto result_martix =
      numeric::ublas::prod(ones_martirx, sample_martrix) / iterations;
    // cout << result_martix << endl;

    // Save the result in csv
    file_name = "./LocalTempData/gibbsResults_" + graph_type + ".csv";
    ofstream file(file_name);
    for (int i = 0; i < my_net.ReturnNumberOfNodes() - 1; i++) {
        file << result_martix(0, i) << ",";
    }
    file << result_martix(0, my_net.ReturnNumberOfNodes() - 1);
    file.close();

    return 0;
}
