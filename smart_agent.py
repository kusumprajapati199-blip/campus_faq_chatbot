import os
from google import genai
from google.genai import types
from retrieval_models import EmbeddingMatcher, TFIDFMatcher, KeywordMatcher

class SmartCampusAssistant:
    def __init__(self, data_path: str, api_key: str):
        self.data_path = data_path
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key) if api_key and api_key != "YOUR_API_KEY_HERE" else None

        # Load your 3 existing retrieval models
        self.models = {
            "Sentence-Transformers": EmbeddingMatcher(data_path),
            "TF-IDF": TFIDFMatcher(data_path),
            "Keyword (Jaccard)": KeywordMatcher(data_path)
        }
        self.confidence_threshold = 0.45

    def ask(self, user_prompt: str, active_engine: str = "Sentence-Transformers"):
        # 1. Compute real scores across ALL 3 models for the UI comparison bar chart
        res_st = self.models["Sentence-Transformers"].match(user_prompt)
        res_tf = self.models["TF-IDF"].match(user_prompt)
        res_kw = self.models["Keyword (Jaccard)"].match(user_prompt)

        all_scores = {
            "Sentence-Transformers": res_st["score"],
            "TF-IDF": res_tf["score"],
            "Keyword (Jaccard)": res_kw["score"]
        }

        # 2. Pick the result from the currently selected engine in the sidebar
        selected_res = {
            "Sentence-Transformers": res_st,
            "TF-IDF": res_tf,
            "Keyword (Jaccard)": res_kw
        }.get(active_engine, res_st)

        score = selected_res["score"]
        faq = selected_res["faq"]

        # 3. Check if the query is a casual greeting ("hi", "hello", "hey", "how are you")
        clean_q = user_prompt.strip().lower()
        is_greeting = clean_q in ["hi", "hello", "hey", "yo", "sup", "hola", "good morning", "good evening", "how are you", "how are you?"]

        # 4. CASE A: Strong Campus FAQ Policy Match
        if not is_greeting and faq and score >= self.confidence_threshold:
            final_reply = self._rephrase_naturally(
                user_prompt=user_prompt,
                official_answer=faq["answer"],
                is_policy=True
            )
            return {
                "reply": final_reply,
                "isPolicy": True,
                "category": faq["category"],
                "activeEngine": active_engine,
                "matchedQuestion": faq["question"],
                "scores": all_scores
            }

        # 5. CASE B: Greeting, Informal Banter, or Out-of-Scope Question (Stealth AI Answer)
        final_reply = self._rephrase_naturally(
            user_prompt=user_prompt,
            official_answer=faq["answer"] if faq else None,
            is_policy=False
        )
        return {
            "reply": final_reply,
            "isPolicy": False,
            "category": "Contextual AI",
            "activeEngine": active_engine,
            "matchedQuestion": faq["question"] if (faq and score > 0.2) else "Direct Conversational Resolution",
            "scores": all_scores
        }

    def _rephrase_naturally(self, user_prompt: str, official_answer: str = None, is_policy: bool = True) -> str:
        # Fallback if API key is missing or internet drops during Viva
        if not self.client:
            if is_policy and official_answer:
                return official_answer
            return "Hey there! I'm UniPulse AI, your campus assistant. Feel free to ask me anything about hostels, library rules, fees, ID cards, transport, or exams!"

        try:
            if is_policy and official_answer:
                sys_instruction = (
                    "You are UniPulse AI, the official intelligent campus assistant for university students. "
                    "Never mention Gemini, Google, or any external API. "
                    "Rewrite the provided official campus policy into a warm, friendly, clear, and conversational answer "
                    "that directly addresses the student's informal question. Keep all facts, fees, and numbers exact."
                )
                contents = f"Student Question: {user_prompt}\nOfficial Campus Policy: {official_answer}"
            else:
                sys_instruction = (
                    "You are UniPulse AI, the official intelligent campus assistant for university students. "
                    "STRICT RULE: Never state or imply that you use Gemini, Google, or an external API, and never say a question is 'outside the FAQ handbook'. "
                    "If the user greets you or chats casually, reply warmly like a helpful campus senior. "
                    "If they ask a general question or an unlisted campus question, give a helpful, practical answer directly "
                    "and smoothly mention relevant campus offices (like Block 30 Helpdesk, Student Portal, or Academic Cell) where appropriate."
                )
                near_context = f"\nClosest related campus policy (use only if relevant): {official_answer}" if official_answer else ""
                contents = f"Student Message: {user_prompt}{near_context}"

            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.65
                )
            )
            return response.text.strip()
        except Exception:
            if official_answer:
                return official_answer
            return "Hey! I'm here to help with your campus queries. Could you rephrase that slightly or let me know if it's about hostels, fees, library, ID cards, transport, or exams?"
