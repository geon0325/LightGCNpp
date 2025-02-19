from data.loader import FileIO


class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config.training_set, config.model_type)
        self.valid_data = FileIO.load_data_set(config.valid_set, config.model_type)
        self.test_data = FileIO.load_data_set(config.test_set, config.model_type)

        self.kwargs = {}
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.'+ self.config.model_type +'.' + self.config.model_name + ' import ' + self.config.model_name
        exec(import_str)
        recommender = self.config.model_name + '(self.config,self.training_data,self.valid_data,self.test_data,**self.kwargs)'
        eval(recommender).execute()
