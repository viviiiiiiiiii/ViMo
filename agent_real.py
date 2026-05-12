

import base64
import torch
import re
from PIL import Image
from typing import Optional, List

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 📍 Import dal core (sempre validi)
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

# 📍 Import per AgentExecutor e ReAct (Versione 2026 / Classic)
# Proviamo i due percorsi più probabili per la v1.2.15
try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

# Importiamo l'intero modulo per accedere alle variabili globali aggiornate
import tools_real 
from load_config import load_config
from Qwen_retrieval import generate_answer


# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# ==========================================
# L'ADATTATORE QWEN (Corretto con Freno a Mano)
# ==========================================
class QwenServerLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
            # 🚀 FIX: Aggiungiamo il System Prompt per uccidere la "personalità" di Qwen
            messages = [
                {"role": "system", "content": "You are a rigid, robotic backend system. You MUST communicate ONLY using the requested ReAct format (Thought, Action, Action Input). NO conversational filler, NO greetings, NO explanations of your plan."},
                {"role": "user", "content": prompt}
            ]
            
            if tools_real.qwen_model is None or tools_real.qwen_processor is None:
                raise ValueError("Errore: I motori del server non sono stati accesi!")

            # 1. Chiamiamo la generazione SENZA stop_words (che causava l'errore)
            risposta = generate_answer(
                        tools_real.qwen_model, 
                        tools_real.qwen_processor, 
                        messages,
                        repetition_penalty=1.15,  # 📍 Abbassato a 1.15 (1.5 è troppo aggressivo)
                        max_new_tokens=256       
                    )

            
            # 2. Gestiamo gli STOP WORDS manualmente qui (Freno a mano software)
            # Se Qwen prova a scrivere "Observation:" da solo, noi lo tagliamo fuori.
            manual_stops = (stop or []) + ["Observation:", "Observation", "\nObservation:"]
            
            for stop_word in manual_stops:
                if stop_word in risposta:
                    # Teniamo solo quello che c'è PRIMA della parola di stop
                    risposta = risposta.split(stop_word)[0]

            return risposta.strip()


# ==========================================
# SETUP AGENTE (Globali) - VERSIONE CORRETTA
# ==========================================

template_universale = """You are an investigative research assistant. Your goal is to provide accurate and complete answers by critically evaluating the information you find.

TOOLS AVAILABLE:
{tools}

RULES OF ENGAGEMENT:
1. ALWAYS evaluate the 'Observation'. Ask yourself: "Does this actually answer the user's question or is it irrelevant?"
2. CROSS-REFERENCE: if the visual tool identifies something that doesn't match the user's context, do NOT stop. Use the textual tool.
3. NO LOOPS: do not repeat the same Action with the same Action Input.
4. MULTI-STEP: you can use tools multiple times to build a complete answer.
5. ONLY ONE MOVE: You must choose EITHER an Action OR a Final Answer. NEVER write both in the same response!
6. SKIP UNNECESSARY ACTIONS: If the Observation from the first tool gives you ALL the information you need (e.g., the name of the painting AND the inventions), DO NOT use the text tool. Go directly to Thought and Final Answer.
7. STOP WRITING: If you choose an Action, you MUST stop generating text immediately after writing the Action Input. Do not hallucinate the next Thought.

🚀 CRITICAL RULE FOR YOUR OUTPUT:
You MUST output ONLY the Action and Action Input. Do NOT write conversational text like "I will start by...". Do NOT explain your plan. Stop generating immediately after writing the Action Input!

MANDATORY FORMAT:
Thought: [Your detailed reasoning about what you have and what you still need]
Action: [{tool_names}]
Action Input: [The specific query for the tool]
Observation: [Result from the tool]

... (Repeat Thought/Action/Action Input/Observation if the information is incomplete)

Thought: I have verified all data and it is complete.
Final Answer: [Summarize only the verified facts that directly answer the question]

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=template_universale,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

vero_qwen = QwenServerLLM()

# Inizializziamo l'agente e l'esecutore
agente = create_react_agent(vero_qwen, tools_real.miei_tools_reali, prompt)
esecutore = AgentExecutor(
    agent=agente, 
    tools=tools_real.miei_tools_reali, 
    verbose=True,
    handle_parsing_errors="Check your output format! Remember to use Action: and Action Input:.",
    max_iterations=5, # Per evitare loop infiniti
    early_stopping_method='force'
)

# ==========================================
# MAIN EXECUTION    
# ==========================================
if __name__ == "__main__":
    print("🚀 Inizializzazione sistema sul server...")
    
    # 1. Caricamento configurazione
    config_dict = load_config()

    class CostruttoreArgs:
        pass
    
    args = CostruttoreArgs()
    for key, value in config_dict.items():
        setattr(args, key, str(value))
    args.top_k = 3
    
    # 2. ACCENSIONE MOTORI (Popola tools_real.qwen_model, ecc.)
    tools_real.start_motors(args)
    
    # 3. TEST AGENTE
# 3. TEST AGENTE SOLO TESTO
percorso_immagine = "foto_buia.jpg"
input_semplice = "Identifica chi ha dipinto 'foto_buia.jpg'. Una volta capito chi è, usa la ricerca testuale per dirmi quali sono le sue invenzioni citate nel database che NON siano quadri."
esecutore.invoke({"input": input_semplice})