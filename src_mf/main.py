from model import ANSModel
from utils import parameter_parser, args_printer, sign_prediction_printer, node_classification_printer
import time


if __name__ == "__main__":
    start = time.time()
    args = parameter_parser()
    ans = ANSModel(args)
    ans.calculate()
    end = time.time()
    ans.save_emb()
    print('Total time: ', end - start)
