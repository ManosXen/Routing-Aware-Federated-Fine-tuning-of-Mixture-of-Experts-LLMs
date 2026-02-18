class NoTrainableParams(Exception):

    def __init__(self):
        self.message = 'No trainable params in the this client. Skipping this round'
        super().__init__(self.message)
