class Data(object):
    def __init__(self, conf, training, valid, test):
        self.config = conf
        self.training_data = training
        self.valid_data = valid
        self.test_data = test #can also be validation set if the input is for validation







