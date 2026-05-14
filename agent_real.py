import base64
import torch
import re
import os
from PIL import Image
from typing import Optional, List

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

import tools_real 
from load_config import load_config
from Qwen_retrieval import generate_answer

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


class QwenServerLLM(LLM):
    current_image_path: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom-multimodal"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
            
            user_content = []
            
            # 1. TRUCCO DEL TAG: Intercettiamo l'immagine nascosta nel prompt di LangChain
            match = re.search(r'\[IMG\](.*?)\[/IMG\]', prompt)
            if match:
                image_path = match.group(1).strip()
                if os.path.exists(image_path):
                    user_content.append({"type": "image", "image": image_path})
                # Rimuoviamo il tag dal testo per non confondere Qwen
                prompt = re.sub(r'\[IMG\].*?\[/IMG\]\n?', '', prompt)
            # Fallback di sicurezza se usi il vecchio metodo
            elif self.current_image_path and os.path.exists(self.current_image_path):
                user_content.append({"type": "image", "image": self.current_image_path})
            
            # 2. Immagini wiki scaricate al volo dal tool
            wiki_images = re.findall(r'\[IMG_WIKI:\s*(.*?)\]', prompt)
            for img_path in set(wiki_images): 
                if os.path.exists(img_path):
                    user_content.append({"type": "image", "image": img_path})
                    print(f"👁️ Qwen sta guardando un'immagine extra da Wikipedia: {img_path}")
            
            user_content.append({"type": "text", "text": prompt})

            messages = [
                {"role": "system", "content": "You are a rigid, robotic backend system. You MUST communicate ONLY using the requested ReAct format (Thought, Action, Action Input). NO conversational filler."},
                {"role": "user", "content": user_content}
            ]
            
            if tools_real.qwen_model is None or tools_real.qwen_processor is None:
                raise ValueError("Errore: I motori del server non sono stati accesi!")

            risposta = generate_answer(
                        tools_real.qwen_model, 
                        tools_real.qwen_processor, 
                        messages,
                        #repetition_penalty=1.15,
                        max_new_tokens=512 # Aumentato per evitare tagli a metà frase
                    )

            manual_stops = (stop or []) + ["Observation:", "Observation", "\nObservation:"]
            for stop_word in manual_stops:
                if stop_word in risposta:
                    risposta = risposta.split(stop_word)[0]

            return risposta.strip()


# ==========================================
# SETUP AGENTE - NUOVO PROMPT A DUE FASI
# ==========================================

template_universale = """Sei un Agente QA Multimodale rigorosissimo. PUOI vedere l'immagine allegata.
NON sei un chatbot. DEVI usare SEMPRE E SOLO questo formato esatto. Le parole chiave del formato (Thought, Action, Action Input, Observation, Final Answer) devono rimanere in inglese.

STRUMENTI A DISPOSIZIONE:
{tools}

REGOLE VITALI:
1. NON INIZIARE MAI A PARLARE SENZA LA PAROLA "Thought:".
2. VALUTAZIONE AD ALTA VOCE: Prima di usare 'tool_leggi_sezione', nel tuo 'Thought' DEVI leggere ad alta voce i titoli dei documenti trovati e spiegare perché ne stai scegliendo uno.
3. ANTI-LOOP (CRITICO): Se hai già letto la Sezione 4 di un documento, NON PUOI leggerla di nuovo. Scegli un'altra sezione o dai la risposta finale.
4. CONOSCENZA INTERNA: Una volta trovato il creatore dell'opera nei documenti, usa la tua intelligenza interna per elencare le sue invenzioni. NON usare i tool per le invenzioni.

FORMATO OBBLIGATORIO:
Thought: 
1) Analisi Visiva: [Descrivi l'immagine]
2) Valutazione Opzioni: [Elenca i documenti trovati e scegli il migliore. Es: "Ho trovato A, B e C. Scelgo A perché..."]
3) Azione: [Cosa farò adesso]
Action: [{tool_names}]
Action Input: [L'input esatto]
Observation: [Risultato dal tool]

... (Ripeti finché non hai la risposta, senza MAI ripetere la stessa lettura)

Thought: Ho identificato l'autore e posso usare la mia conoscenza interna per le invenzioni.
Final Answer: [La tua risposta finale]

Inizia!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=template_universale,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

vero_qwen = QwenServerLLM()

agente = create_react_agent(vero_qwen, tools_real.miei_tools_reali, prompt)
esecutore = AgentExecutor(
    agent=agente, 
    tools=tools_real.miei_tools_reali, 
    verbose=True,
    handle_parsing_errors="Check your output format! Remember to use Action: and Action Input:.",
    max_iterations=6, 
    early_stopping_method='force'
)

def run_agentic_rag(image_path, question):
    vero_qwen.current_image_path = image_path
    try:
        result = esecutore.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Errore Agente: {str(e)}"

if __name__ == "__main__":
    print("🚀 Inizializzazione sistema sul server...")
    
    config_dict = load_config()

    class CostruttoreArgs: pass
    args = CostruttoreArgs()
    for key, value in config_dict.items():
        setattr(args, key, str(value))
    args.top_k = 3
    
    tools_real.start_motors(args)
    
    percorso_immagine = "foto_buia.jpg"
    vero_qwen.current_image_path = percorso_immagine


# INIETTIAMO IL TAG [IMG] E GLI DIAMO ISTRUZIONI PRECISE SUL FILE E SULLE INVENZIONI
    input_semplice = f"[IMG]{percorso_immagine}[/IMG]\nGuarda l'immagine allegata. Il nome del file è '{percorso_immagine}'. Usa 'tool_ricerca_visiva' passando '{percorso_immagine}' come Action Input per trovare i documenti. Leggi i documenti per scoprire chi è l'autore dell'opera. Una volta scoperto l'autore, usa la tua conoscenza interna per elencare le SUE invenzioni famose (non cercare le invenzioni nei documenti)."

    print(f"\n🧠 Avvio indagine di Qwen. Occhi puntati su: {percorso_immagine}...")
    
    # Salva e stampa a caratteri cubitali per forzare la scrittura nel log di SLURM!
    risultato_finale = esecutore.invoke({"input": input_semplice})
    
    print("\n" + "🔥"*25)
    print("🎯 RISPOSTA FINALE DELL'AGENTE:")
    if isinstance(risultato_finale, dict) and "output" in risultato_finale:
        print(risultato_finale["output"])
    else:
        print(risultato_finale)
    print("🔥"*25 + "\n")