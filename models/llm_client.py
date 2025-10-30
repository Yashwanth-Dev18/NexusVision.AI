from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class FlanT5Client:
    def __init__(self):
        self.model_name = "google/flan-t5-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        print("Model loaded successfully!")
    
    def generate_response(self, prompt, max_length=512):
        try:
            # FLAN-T5 works better with instruction prefix
            input_text = f"Answer this question based on the context: {prompt}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"