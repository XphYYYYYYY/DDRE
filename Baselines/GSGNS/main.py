from model import AdversarialNegativeSampling
from utils import args_parser


args = args_parser()
gsgns = AdversarialNegativeSampling(args)
gsgns.fit()
gsgns.save_emb()
