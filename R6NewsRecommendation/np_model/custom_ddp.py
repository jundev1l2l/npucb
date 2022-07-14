from torch.nn.parallel import DistributedDataParallel as DDP


class CustomDDP(DDP):
    def __init__(self, *args, **kwargs):
        super(CustomDDP, self).__init__(*args, **kwargs)
        self.name = self.module.name

    def predict(self, *args, **kwargs):
        return self.module.predict(*args, **kwargs)
