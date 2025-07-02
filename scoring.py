from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Model:
    def __init__(self):
        model_id = "OpenAssistant/oasst-sft-1-pythia-12b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()

    def predict(self, request):
        prompt = request.get("input", "")

        if not prompt.strip():
            return {"output": "No input provided."}

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"output": decoded}
