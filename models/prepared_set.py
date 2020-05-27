class PreparedSet:
    blog_type_array = 0
    features_array = []
    features_labels_array = []
    class_names = ["Diarystyka", "Refleksja", "Krytyka", "Informacja", "Porada", "Modelowanie", "Fikcjonalność",
                   "Nieblog", "Filtr", "Inne"]

    def __init__(self, blog_type_array, features_array, labels_array):
        self.blog_type_array = blog_type_array
        self.features_array = features_array
        self.features_labels_array = labels_array

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])



    def get_blog_types(self):
        return self.blog_type_array

    def get_blog_type(self, i):
        return self.blog_type_array[i]

    def get_blog_feature(self, i):
        return self.features_array[i]

    def get_features(self):
        return self.features_array

    def get_class_names(self):
        return self.class_names

    def get_class_name(self, i):
        return self.class_names[i]

    def get_features_labels(self):
        return self.features_labels_array

    def get_feature_label(self, i):
        return self.features_labels_array[i]
