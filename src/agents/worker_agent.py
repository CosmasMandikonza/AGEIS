# src/agents/worker_agent.py
import json
import logging
import ollama
from typing import Dict, Optional, List
from bee_agent_framework import Agent, Tool, Message
from config.config import Config
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

class ComplianceAnalysisTool(Tool):
    """Tool for compliance analysis using RAG."""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(
            name="compliance_check",
            description="Check text for compliance violations against regulations"
        )
        self.vector_store = vector_store
    
    async def run(self, query: str) -> str:
        """Run compliance check on the query."""
        context = self.vector_store.get_context_for_query(query)
        return context

class WorkerAgent:
    """Primary analysis agent using Granite 4.0 Tiny."""
    
    def __init__(self, vector_store: VectorStore):
        self.config = Config()
        self.vector_store = vector_store
        self.ollama_client = ollama.Client(host=self.config.OLLAMA_HOST)
        self.conversation_history = []
        
        # Initialize BeeAI agent
        self.compliance_tool = ComplianceAnalysisTool(vector_store)
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create BeeAI agent with tools."""
        system_prompt = """You are an expert compliance co-pilot. Your task is to analyze conversation snippets in the context of relevant financial regulations. 

Your responsibilities:
1. Identify potential compliance violations
2. Explain risks clearly and concisely
3. Provide constructive, alternative phrasing
4. Be helpful, not punitive

Output format: Always respond with valid JSON containing:
{
    "risk_detected": boolean,
    "explanation": "Clear explanation if risk detected",
    "suggestion": "Alternative compliant phrasing if risk detected"
}"""
        
        # Create agent with Granite 4.0 Tiny
        agent = Agent(
            model_name=self.config.LOCAL_MODEL,
            system_prompt=system_prompt,
            tools=[self.compliance_tool],
            ollama_client=self.ollama_client
        )
        
        return agent
    
    async def analyze_utterance(self, utterance: str, context: List[str] = None) -> Dict[str, any]:
        """
        Analyze an utterance for compliance risks.
        
        Args:
            utterance: The current statement to analyze
            context: Previous conversation context
            
        Returns:
            Analysis result dictionary
        """
        try:
            # Build conversation context
            context_str = ""
            if context and len(context) > 0:
                context_str = "\n".join(context[-3:])  # Last 3 exchanges
            
            # Get relevant compliance rules
            compliance_context = self.vector_store.get_context_for_query(utterance)
            
            # Construct prompt
            prompt = f"""
Relevant Compliance Rules:
{compliance_context}

---
Conversation History (for context):
{context_str}

---
Current Utterance to Analyze:
"{utterance}"

---
Analysis Request:
Based on the rules and the conversation, does the "Current Utterance" pose a compliance risk? 
Provide your analysis in the specified JSON format.
"""
            
            # Use Ollama directly for more control
            response = self.ollama_client.generate(
                model=self.config.LOCAL_MODEL,
                prompt=prompt,
                system=self._get_system_prompt(),
                format='json'
            )
            
            # Parse response
            try:
                result = json.loads(response['response'])
                
                # Ensure all required fields exist
                if 'risk_detected' not in result:
                    result['risk_detected'] = False
                if 'explanation' not in result:
                    result['explanation'] = ""
                if 'suggestion' not in result:
                    result['suggestion'] = ""
                
                return result
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {response['response']}")
                return {
                    "risk_detected": False,
                    "explanation": "Analysis error",
                    "suggestion": ""
                }
            
        except Exception as e:
            logger.error(f"Error in compliance analysis: {e}")
            return {
                "risk_detected": False,
                "explanation": f"Analysis error: {str(e)}",
                "suggestion": ""
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the model."""
        return """You are an expert compliance co-pilot for financial services. 
Analyze conversations for regulatory compliance risks.
Always respond in valid JSON format with keys: risk_detected (boolean), explanation (string), suggestion (string).
Be helpful and constructive in your suggestions."""
    
    def update_conversation_history(self, utterance: str):
        """Update the conversation history."""
        self.conversation_history.append(utterance)
        # Keep only last 10 utterances
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

---
# src/agents/guardian_agent.py
import json
import logging
import ollama
from typing import Dict, Optional
from config.config import Config

logger = logging.getLogger(__name__)

class GuardianAgent:
    """Safety and quality assurance agent using granite-guardian."""
    
    def __init__(self):
        self.config = Config()
        self.ollama_client = ollama.Client(host=self.config.OLLAMA_HOST)
    
    async def review_alert(self, alert: Dict[str, any], original_utterance: str) -> Dict[str, any]:
        """
        Review and refine an alert from the WorkerAgent.
        
        Args:
            alert: Alert dictionary from WorkerAgent
            original_utterance: The original statement being analyzed
            
        Returns:
            Refined alert dictionary
        """
        try:
            # Only review if risk was detected
            if not alert.get('risk_detected', False):
                return alert
            
            # Construct review prompt
            prompt = f"""
You are a safety and quality reviewer for compliance alerts. Review the following alert:

Original Statement: "{original_utterance}"
Alert Explanation: "{alert['explanation']}"
Suggested Alternative: "{alert['suggestion']}"

Your tasks:
1. Ensure the alert is clear, professional, and helpful
2. Check that the suggested alternative doesn't introduce new risks
3. Refine the language to be constructive, not punitive
4. Verify the suggestion is actionable

Provide your refined alert in JSON format:
{{
    "risk_detected": true,
    "explanation": "Refined explanation",
    "suggestion": "Refined suggestion",
    "quality_score": 0-10
}}
"""
            
            # Use guardian model for review
            response = self.ollama_client.generate(
                model=self.config.GUARDIAN_MODEL,
                prompt=prompt,
                system=self._get_system_prompt(),
                format='json'
            )
            
            # Parse response
            try:
                refined = json.loads(response['response'])
                
                # Merge with original alert
                result = alert.copy()
                
                # Update with refined content if quality is good
                quality_score = refined.get('quality_score', 5)
                if quality_score >= 7:
                    result['explanation'] = refined.get('explanation', alert['explanation'])
                    result['suggestion'] = refined.get('suggestion', alert['suggestion'])
                    result['guardian_reviewed'] = True
                    result['quality_score'] = quality_score
                else:
                    # If quality is low, keep original but flag it
                    result['guardian_reviewed'] = True
                    result['quality_score'] = quality_score
                    result['needs_improvement'] = True
                
                return result
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse guardian response: {response['response']}")
                # Return original alert if parsing fails
                alert['guardian_reviewed'] = False
                return alert
            
        except Exception as e:
            logger.error(f"Error in guardian review: {e}")
            alert['guardian_reviewed'] = False
            return alert
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the guardian model."""
        return """You are a quality assurance specialist for compliance alerts.
Your role is to ensure alerts are:
- Clear and easy to understand
- Professional and respectful in tone
- Constructive and helpful
- Free from unintended risks or ambiguity
Always respond in JSON format.
Rate quality from 0-10, where 10 is perfect."""
    
    async def validate_compliance_rules(self, rules: List[str]) -> List[str]:
        """
        Validate that compliance rules are appropriate and safe.
        
        Args:
            rules: List of compliance rules
            
        Returns:
            Validated rules
        """
        # This method can be used to verify that the RAG-retrieved rules
        # are appropriate and don't contain any harmful content
        validated_rules = []
        
        for rule in rules:
            if self._is_rule_safe(rule):
                validated_rules.append(rule)
        
        return validated_rules
    
    def _is_rule_safe(self, rule: str) -> bool:
        """Check if a rule is safe to use."""
        # Basic safety checks
        unsafe_patterns = [
            'discriminat',
            'personal information',
            'confidential'
        ]
        
        rule_lower = rule.lower()
        for pattern in unsafe_patterns:
            if pattern in rule_lower:
                logger.warning(f"Potentially unsafe rule detected: {pattern}")
                return False
        
        return True