class HyperparameterContainer:

    def __init__(self):
        self.learning_rate = {
            "1": 0.005,
            "2": 0.0005,
            "3": 0.00005,
            "4": 0.000005
        }
        self.dropout = {
            "1": 0.,
            "2": 0.2
        }
        self.l1_coef = {
            "1": 0.,
            "2": 1e-6,
            "3": 1e-3,
            "4": 1e0
        }
        self.l2_coef = {
            "1": 0.,
            "2": 1e-6,
            "3": 1e-3,
            "4": 1e0
        }
