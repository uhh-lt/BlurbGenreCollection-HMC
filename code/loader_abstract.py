class Loader_interface:


    """
    Loads text and labels of a dataset
    @return: A list consisting of three collections [train, dev, test].
        Each collection contains a list of tuples. Each pair consist of a text and the corresponding set of labels.
    """
    def load_data_multiLabel(self): raise NotImplementedError



    """
    Loads relations of the genre Hierarchy
    @return: Tuple of relation-set and set of parents without children
    The relation-set consist of relation-pairs with the first entry being the parent, and second entry being the child
    The second set consists of genres that have neither a parent nor a child
    """
    def read_relations(self): raise NotImplementedError


    """
    Loads the co-occurences of a dataset, by descending order of their frequency
    @return: A Tuple of
    """
    def read_all_genres(self): raise NotImplementedError
