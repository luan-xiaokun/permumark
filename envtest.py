from copy import deepcopy

from transformers import AutoModelForCausalLM
from permumark import PermutationWatermark

model = AutoModelForCausalLM.from_pretrained(
    "models/meta-llama/Llama-3.2-1B", trust_remote_code=True
)
source = deepcopy(model)
pw = PermutationWatermark(model.config)
identity = pw.generate_random_identity()
insertion_result = pw.insert_watermark(model, identity)

extract_res = pw.extract_watermark(source, model)
print(f"Extracted identity matches: {extract_res.identity == identity}")
