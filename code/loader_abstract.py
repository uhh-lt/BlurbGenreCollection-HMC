class Loader_Interface:


    """
    Loads text and labels of a dataset
    @return: A list consisting of three collections [train, dev, test].
        Each collection contains a list of tuples. Each pair consist of a text (String) and the corresponding set of labels (Set of Strings).
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
    Requires load_data_multiLabel
    """
    def read_all_genres(self):
        occurences = []
        frequency = []
        hierarchy = set([])
        co_occurences_path =  os.path.join(os.path.dirname(__file__),
         '../resources', 'co_occurences')
        if os.path.exists(co_occurences_path):
            co_occurences_file = open(co_occurences_path, 'rb')
            occurences, frequency = pickle.load(co_occurences_file)
        else:
            train, dev, test = self.load_data_multiLabel()
            all_data = train + dev + test
            for entry in all_data:
                genres = entry[1]
                if genres in occurences:
                    frequency[occurences.index(genres)] +=1
                else:
                    occurences.append(genres)
                    frequency.append(1)
            co_occurence_file = open(co_occurences_path, 'wb')
            pickle.dump([occurences,frequency], co_occurence_file)

        occurences = zip(occurences, frequency)
        occurences = sorted(occurences, key=operator.itemgetter(1), reverse = True)

        return [hierarchy, occurences]
