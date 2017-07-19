class ExponentialMovingAverage:
    def __init__(self, decay=0.95):
        self._init = False
        self._ma = 0.
        self._decay = decay
        self._iter = 0
        self._decay_iter = 1. / (1-self._decay)

    @property
    def moving_average(self):
        return self._ma

    def update(self, value):
        self._iter += 1
        if self._init:
            if self._iter < self._decay_iter:
                self._ma -= 1. / self._iter * (self._ma - value)
            else:
                self._ma -= (1 - self._decay) * (self._ma - value)
        else:
            self._ma = value
            self._init = True

    def reset(self):
        self._iter = 0
        self._init = False
