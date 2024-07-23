import os
from dotenv import load_dotenv

from openai import OpenAI
from httpx import Timeout

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model

@register_model("llama_405b_instruct")
class Llama405B_instruct(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "accounts/fireworks/models/llama-v3p1-405b-instruct"

        load_dotenv()
        self.client = OpenAI(
            base_url = "https://api.fireworks.ai/inference/v1",
            api_key=os.getenv('FIREWORKS_API_KEY'),
            timeout=Timeout(timeout=100.0, connect=20.0)
        )
        
        
