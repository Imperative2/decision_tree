
from sklearn import tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # <- path  to open graphviz to visualize data
import graphviz
import pandas as pandas
import numpy as numpy
import csv

from os import walk

from utils import LoadRawData
from utils import SplitDataset
from algorithms import DecisionTree
from utils import FileWriter


def get_current_test_file_path():
    return os.getcwd() + "\\test_data"


def get_test_file_list(test_file_path):
    print(test_file_path)
    f = []
    for (dirpath, dirnames, filenames) in walk(test_file_path):
        f.extend(filenames)
        break
    return f



# criterion - parameter :  The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
criterions_array = ["gini" , "entropy" ]
# splitter - parameter : The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
spliters_array = ["best", "random"]

file_writer = FileWriter("C:\\aa.txt")

# load all files
files_loaded = get_test_file_list(get_current_test_file_path())

print(get_current_test_file_path())
print(get_test_file_list(get_current_test_file_path()))

#for each file run experiment
for file in files_loaded:
    # load data from file
    file_name = file
    file_path = os.getcwd() + "\\test_data\\" + file_name

    print(file_name)

    raw_data_loader = LoadRawData(file_path)
    raw_data = raw_data_loader.load_raw_dataset()

    # split blogs into training and test sets
    training_set , test_set = SplitDataset.split_sets(raw_data)

    for criterion in criterions_array:
        for spliter in spliters_array:
            print("Running alg for: ",file_name, criterion, spliter)
            #create algorithm
            classifier = DecisionTree(criterion, spliter, training_set, test_set)

            # train decision tree
            classifier.train()

            # predict result and caluculate quality of clasiffier
            result = classifier.test()
            print("result: ",result)

            # safe result to txt
            res_str = file_name+" | "+criterion+" | "+spliter+" | "+str(result)+"\n"
            file_writer.write(res_str)


# close write file
file_writer.close()








#

#
#
# def load_dataset(filename):
#     with open(filename, newline="") as csv_file:
#         df = pandas.read_csv(csv_file, delimiter=",", skiprows=1 )
#         dataset = df.values.tolist()
#     print("\n----------Dataset------------\n")
#  #   print(dataset)
#     return dataset
#
# def load_classes(dataset):
#     Y = []
#     print(dataset)
#     for row in dataset:
#         Y.append(row[0])
#
#     print("\n----------Y------------\n")
#  #   print(Y)
#     return Y
#
# def load_features(filename):
#     with open(filename, newline="") as csv_file:
#         df = pandas.read_csv(csv_file, delimiter=",", skiprows=1 )
#         X = df.values.tolist()
#     print("\n----------X------------\n")
# #    print(X)
#     return X
#
# def load_feature_labels(filename):
#     with open(filename, newline="") as csv_file:
#         reader = csv.reader(csv_file, delimiter=',')
#         for row in reader:
#             labels = row
#             break;
#     return labels
#
#
# def generate_graph(dot_data, filename,result_path):
#     graph = graphviz.Source(dot_data)
#     graph.render(result_path+"\\result_"+filename)
#
#
#
#
# #------------------------------ ALG START ------------------------------------------------
#
#
# class_names = ["Diarystyka","Refleksja","Krytyka","Informacja","Porada","Modelowanie","Fikcjonalność","Nieblog","Filtr","Inne"]
#
# test_file_path = get_current_test_file_path()
# result_dir_path = get_current_result_dir()
# print(test_file_path)
# file_list = get_test_file_list(test_file_path)
# print(file_list)
#
# print("loading main data set")
# dataset = load_dataset("dataset.csv")
# print("loading class list")
# Y = load_classes(dataset)
#
# for file in file_list:
#     print("loading data for file "+file)
#     X = load_features(test_file_path+"\\"+file)
#     feature_labels = load_feature_labels(test_file_path+"\\"+file)
#     print("--------------Feature Labels------------")
#     print(feature_labels)
#     print("training for "+file)
#     clf = tree.DecisionTreeClassifier()
#     clf = clf.fit(X,Y)
#     print("trained")
#     print("generating graph for "+file)
#     generate_graph(tree.export_graphviz(clf, out_file=None,
#                                     filled=True, rounded=True,
#                                     class_names = class_names,
#                                     feature_names=feature_labels,
#                                     special_characters=True), file, result_dir_path)
#
#
#
#
#


