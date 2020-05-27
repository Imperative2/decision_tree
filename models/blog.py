class Blog:
    id =0
    blog_type = 0
    features = []

    def __init__(self, id, disease_class, features):
        self.id = id
        self.disease_class = disease_class
        self.features = features

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

    def getId(self):
        return self.id

    def getBlogType(self):
        return self.blog_type

    def get_features(self):
        return self.features
