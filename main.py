from para import Parameter
from train import Trainer

if __name__ == '__main__':
    args = Parameter().args
    trainer = Trainer(args)
    trainer.run()
