import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOllama
from langchain import PromptTemplate, LLMChain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
model = OllamaLLM(model="phi4:latest")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens = 128,
    chunk_length_s = 15,
    batch_size = 16,
    return_timestamps = True,
    torch_dtype=torch_dtype,
    device=device,
    
)

audio_directory = "C:/transcriptionenv/Pac Life"

def trancriptor(audio_directory):
    results = []

    for filename in os.listdir(audio_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(audio_directory, filename)
            print(f"Transcribing {filename}...")
        
            # Transcribe audio
            transcription = pipe(file_path)
        
            # Append results to the list
            results.append({"filename": filename, "transcription": transcription['text']})
            
            df = pd.DataFrame(results)
    return df

df = trancriptor(audio_directory)

def Call_Analyser(query):
    prompt = PromptTemplate(
        input_variables=["query", "chain_of_thought"],
        template="""
         You are a customer care data analyst, you have a call recording transcription between an agent and a client. Your task is to extract specific
         statement from transcript that correspond to the following call principles. If a specific type of statment is not present in the transcript, 
         respond with 'Not performed' for that category.
         
         Extract the exact sentence(s) from the transcript for each of the following:
         
         Important Note: "Please do not generate own sentence for the following."
         
         1. The agent's greeting to the client.  
         2. The agent empathizing with the customer's situation. 
         3. The agent asking for identity information to validate the policy number. 
         4. The agent resolving the customer's issue.  
         5. The agent summarizing the problem and solution for the customer.  
         6. The agent asking if the customer needs any additional assistance.  
         7. The agent's closing statement.

         You may extract multiple sentences if they are part of the same action or statement.Format your response in json format as follows:
            {chain_of_thought}
         The below is the transcription:
            {query}        
        """
    )

    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run(query=query, chain_of_thought="""
    You may extract multiple sentences if they are part of the same action or statement.Format your response in json format as follows:
         Greeting: "Your extracted sentence(s) from the transcript Here" 
         Empathy: "Your extracted sentence(s) or "Not performed""
         Identity Verification: "Your extracted sentence(s) or "Not performed"" 
         Issue Resolution: "Your extracted sentence(s) or "Not performed""
         Summary: "Your extracted sentence(s) or "Not performed""
         Additional Assistance: "Your extracted sentence(s) or "Not performed"" 
         Closing: "Your extracted sentence(s)from the transcript"
         
         Confidentiality Notice: Ensure the response refers only to "agent" and "client" without directly referencing their identities. 
          """) # Provide a value for chain_of_thought
    return response.strip()

def result_generator(df):
    res1 = []
    for i in df['transcription']:
        itemname = i
        res1.append(Call_Analyser(itemname))

result_generator(df)