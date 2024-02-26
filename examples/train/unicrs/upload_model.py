import sys

sys.path.append("./src")

from recwizard.modules.unicrs import UnicrsGen, UnicrsRec, UnicrsGenTokenizer, UnicrsRecTokenizer
from recwizard.utility import HF_ORG, load_json_file_from_dataset
import os

dataset = sys.argv[1]

gen_module = UnicrsGen.from_pretrained(f"./save/unicrs_conv/{dataset}/model_best")
rec_module = UnicrsRec.from_pretrained(f"./save/unicrs_rec/{dataset}/model_best")

gen_repo = os.path.join(HF_ORG, f"UnicrsGen-{dataset}")
rec_repo = os.path.join(HF_ORG, f"UnicrsRec-{dataset}")
gen_module.push_to_hub(gen_repo)
rec_module.push_to_hub(rec_repo)

local_dataset = dataset + "_unicrs"

gen_tokenizer = UnicrsGenTokenizer.load_from_dataset(local_dataset)
rec_tokenizer = UnicrsRecTokenizer.load_from_dataset(local_dataset)

gen_tokenizer.push_to_hub(gen_repo)
rec_tokenizer.push_to_hub(rec_repo)
