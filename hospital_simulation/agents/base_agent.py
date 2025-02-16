from typing import Any, Dict, List
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage
import os
import time
from datetime import datetime

class BaseAgent:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.7,
        )
        self.chain = None
        self.prompt = None
        self.model_name = model_name
        self.logs = []

    def _log_step(self, step_type: str, message: str):
        """Log a step with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {step_type}: {message}"
        print(log_entry)
        self.logs.append(log_entry)

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic and logging."""
        self._log_step("MODEL", f"Using {self.model_name}")
        self._log_step("PROMPT", f"Sending prompt:\n{prompt}")
        
        for attempt in range(max_retries):
            try:
                self._log_step("REQUEST", f"Attempt {attempt + 1}/{max_retries}")
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                if response and hasattr(response, 'content'):
                    self._log_step("RESPONSE", f"Received response:\n{response.content}")
                    return response.content
                else:
                    self._log_step("WARNING", "Empty or invalid response from LLM")
                    if attempt < max_retries - 1:
                        continue
            except Exception as e:
                self._log_step("ERROR", f"Error: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self._log_step("RETRY", f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self._log_step("FALLBACK", "Max retries reached. Using fallback response.")
                    return self._get_fallback_response()
        
        self._log_step("FALLBACK", "All attempts failed. Using fallback response.")
        return self._get_fallback_response()

    def _get_fallback_response(self) -> str:
        """Get a fallback response in case of LLM failure."""
        fallback = """I apologize, but I'm unable to provide a detailed response at the moment. 
Please proceed with standard protocols and consider the following general guidelines:
- Monitor patient's vital signs
- Document all symptoms carefully
- Consider basic diagnostic tests
- Consult with colleagues if needed
- Ensure patient comfort and safety"""
        self._log_step("FALLBACK", f"Using fallback response:\n{fallback}")
        return fallback

    def run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Run the agent's chain with the given inputs."""
        if not self.prompt:
            raise ValueError("Prompt not initialized. Call setup_chain first.")
        
        self._log_step("INPUT", f"Received inputs: {inputs}")
        
        # Format the prompt
        formatted_prompt = self.prompt.format(**inputs)
        
        # Call LLM and get response
        response = self._call_llm(formatted_prompt)
        
        # Ensure we have a valid response
        if not response or not isinstance(response, str):
            response = self._get_fallback_response()
        
        result = {"response": response.strip(), "logs": self.logs}
        self._log_step("OUTPUT", f"Final response: {response.strip()}")
        return result 