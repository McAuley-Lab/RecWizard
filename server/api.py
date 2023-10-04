from recwizard import monitoring
from fastapi import FastAPI
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
state = {}
logger.info("Pipeline loaded")
app = FastAPI()
model_list = [
    "Unicrs",
    "Unicrs-redial",
    "Redial",
    "ChatGPT-expansion",
    "ChatGPT-recommendation",
]

from recwizard import (
    FillBlankConfig,
    FillBlankPipeline,
    UnicrsGen,
    UnicrsRec,
    ChatgptGen,
    ChatgptRec,
    ExpansionPipeline,
    ExpansionConfig,
    RedialGen,
    RedialRec,
)

from recwizard.utility import DeviceManager

DeviceManager.initialize("cpu")


@app.get("/")
def read_root():
    start_time = time.time()
    return {"text": "Welcome to Recwizard inference server", "Time": start_time}


@app.get("/listmodels")
def list_models():
    return model_list


@app.get("/loadmodel")
def load_model(model_name):
    try:
        if model_name in state:
            return "success"
        if model_name not in model_list:
            return "failure"
        if model_name == "Unicrs":
            state[model_name] = FillBlankPipeline(
                config=FillBlankConfig(),
                gen_module=UnicrsGen.from_pretrained("recwizard/unicrs-gen-redial"),
                rec_module=UnicrsRec.from_pretrained("recwizard/unicrs-rec-redial"),
            )

        if model_name == "Unicrs-redial":
            state[model_name] = FillBlankPipeline(
                config=FillBlankConfig(),
                gen_module=UnicrsGen.from_pretrained("recwizard/unicrs-gen-redial"),
                rec_module=UnicrsRec.from_pretrained("recwizard/unicrs-rec-redial"),
            )

        if model_name == "Redial":
            state[model_name] = ExpansionPipeline(
                config=ExpansionConfig(),
                gen_module=RedialGen.from_pretrained("recwizard/redial-gen"),
                rec_module=RedialRec.from_pretrained("recwizard/redial-rec"),
            )

        if model_name == "ChatGPT-expansion":
            state[model_name] = ExpansionPipeline(
                config=ExpansionConfig(),
                gen_module=ChatgptGen.from_pretrained(
                    "recwizard/chatgpt-gen-expansion"
                ),
                rec_module=RedialRec.from_pretrained("recwizard/redial-rec"),
                use_rec_logits=False,
            )

        if model_name == "ChatGPT-recommendation":
            state[model_name] = FillBlankPipeline(
                config=FillBlankConfig(),
                gen_module=UnicrsGen.from_pretrained("recwizard/unicrs-gen-redial"),
                rec_module=ChatgptRec.from_pretrained(
                    "recwizard/chatgpt-rec-fillblank"
                ),
            )

        return "success"
    except Exception as e:
        logger.error(e)
        return "failure"


@app.get("/predict")
def predict(
    model_name: str,
    query: str,
    mode: str = "info",
    rec_args: dict = None,
    gen_args: dict = None,
):
    logger.info(f"Query: {query}")
    load_model(model_name)
    with monitoring(mode) as m:
        response = state[model_name].response(
            query, return_dict=True, rec_args=rec_args, gen_args=gen_args
        )
        logger.info(f"Response: {response}")
        return {
            "query": query,
            "response": {"output": response["output"], "links": response["links"]},
            "graph": m.graph,
        }
