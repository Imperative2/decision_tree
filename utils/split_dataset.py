
from sklearn.model_selection import train_test_split
from models import PreparedSet


class SplitDataset:

    @staticmethod
    def split_sets(raw_data):

        features_labels = raw_data[0]
        raw_data = raw_data[1:]

        x_train, x_test = train_test_split(raw_data, test_size=0.3)

        train_set_classes = []
        train_set_features = []

        test_set_classes = []
        test_set_features = []

        for row in x_train:
            train_set_classes.append(row[0])
            train_set_features.append(row[1:])

        for row in x_test:
            test_set_classes.append(row[0])
            test_set_features.append(row[1:])

        train_set = PreparedSet(train_set_classes, train_set_features, features_labels)
        test_set = PreparedSet(test_set_classes, test_set_features, features_labels)

        # print("test_set", test_set.get_blog_types())

        return train_set, test_set
