import os
from dotenv import load_dotenv
from src.models.reasoning_engine import AloraReasoningEngine
from src.utils.knowledge_base import AloraKnowledgeBase
import logging

class AloraAssistant:
    def __init__(self, log_level=logging.INFO):
        """
        Initialize Alora AI Assistant
        
        Args:
            log_level (int): Logging level for the assistant
        """
        # Configure logging
        logging.basicConfig(level=log_level, 
                            format='[Alora] %(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize core components
        self.reasoning_engine = AloraReasoningEngine()
        self.knowledge_base = AloraKnowledgeBase()
        
        self.logger.info("Alora AI Assistant initialized successfully")
        
    def process_input(self, user_input: str) -> str:
        """
        Process user input through Alora's reasoning and knowledge systems
        
        Args:
            user_input (str): Input text from user
        
        Returns:
            str: Generated response
        """
        try:
            # Analyze input semantically
            reasoning_result = self.reasoning_engine.analyze(user_input)
            
            # Retrieve contextual knowledge
            relevant_knowledge = self.knowledge_base.query(user_input)
            
            # Generate response
            response = self.generate_response(reasoning_result, relevant_knowledge)
            
            self.logger.info(f"Processed input: {user_input[:50]}...")
            return response
        
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return "I apologize, but I encountered an error while processing your request."
        
    def generate_response(self, reasoning, knowledge):
        """
        Generate a sophisticated response based on reasoning and knowledge
        
        Args:
            reasoning (dict): Reasoning analysis results
            knowledge (dict): Relevant knowledge retrieved
        
        Returns:
            str: Generated response
        """
        # More advanced response generation
        sentiment = reasoning.get('sentiment', {}).get('label', 'neutral')
        context_details = f"Sentiment: {sentiment}"
        
        response = (f"Alora Analysis: {context_details}\n"
                    f"Insights: {reasoning}\n"
                    f"Contextual Knowledge: {knowledge}")
        
        return response

def main():
    """
    Main function to demonstrate Alora AI Assistant
    """
    alora = AloraAssistant()
    print("Alora AI Assistant activated. Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Alora: Goodbye! Hope to assist you again soon.")
                break
            
            response = alora.process_input(user_input)
            print("Alora:", response)
        
        except KeyboardInterrupt:
            print("\nAlora: Operation cancelled. Goodbye!")
            break

if __name__ == "__main__":
    main()
