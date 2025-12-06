"""
Statement typing module using hybrid rule-based + ML classifier approach.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class StatementTyper:
    """Classifies statements into semantic types."""
    
    # Statement types
    TYPES = [
        "Instruction",
        "Conditional",
        "Warning/Policy",
        "Definition",
        "Assertion/Fact",
        "Meta-safety",
        "Refusal",
        "Hedge/Uncertainty"
    ]
    
    # Rule-based patterns
    TYPE_PATTERNS = {
        "Instruction": [
            r'\b(should|must|need to|ought to|required to|instruct|direct|command)\b',
            r'\b(do|perform|execute|implement|apply)\b',
            r'^(you|one|they|we)\s+(should|must|need|ought)',
        ],
        "Conditional": [
            r'\b(if|when|unless|provided that|assuming|given that)\b',
            r'\b(then|else|otherwise)\b',
            r'\b(depends on|contingent upon|subject to)\b',
        ],
        "Warning/Policy": [
            r'\b(warning|caution|danger|risk|hazard)\b',
            r'\b(policy|regulation|compliance|violation)\b',
            r'\b(prohibited|forbidden|illegal|unlawful)\b',
            r'\b(must comply|must follow|must adhere)\b',
        ],
        "Definition": [
            r'\b(is|are|means|refers to|defined as|denotes)\b',
            r'\b(consists of|comprises|constitutes)\b',
            r'^(a|an|the)\s+\w+\s+(is|are)',
        ],
        "Assertion/Fact": [
            r'\b(is|are|was|were|has|have|had)\b',
            r'\b(according to|research shows|studies indicate)\b',
            r'\b(fact|evidence|data|findings)\b',
        ],
        "Meta-safety": [
            r'\b(cannot|can\'t|unable to|not allowed|not permitted)\b',
            r'\b(should not|must not|ought not)\b',
            r'\b(restricted|limited|constrained)\b',
            r'\b(safety|ethical|responsible|appropriate)\b',
        ],
        "Refusal": [
            r'\b(cannot|can\'t|unable to|refuse|decline)\b',
            r'\b(will not|won\'t|should not|must not)\b',
            r'\b(not provide|not give|not share|not disclose)\b',
            r'\b(not appropriate|not suitable|not allowed)\b',
        ],
        "Hedge/Uncertainty": [
            r'\b(might|may|could|possibly|perhaps|maybe)\b',
            r'\b(potentially|likely|unlikely|probably|possibly)\b',
            r'\b(some|certain|various|several|many)\b',
            r'\b(uncertain|unclear|ambiguous|unpredictable)\b',
            r'\b(depends|varies|fluctuates|changes)\b',
        ]
    }
    
    def __init__(self, use_ml: bool = False, model_name: Optional[str] = None):
        """
        Initialize the typer.
        
        Args:
            use_ml: Whether to use ML classifier (requires model)
            model_name: Optional transformer model for ML classification
        """
        self.use_ml = use_ml
        self.ml_model = None
        self.ml_tokenizer = None
        
        if use_ml and model_name:
            try:
                self.ml_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.ml_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(self.TYPES)
                )
                self.ml_model.eval()
            except Exception as e:
                print(f"Warning: Could not load ML model {model_name}: {e}")
                print("Falling back to rule-based classification only.")
                self.use_ml = False
    
    def classify_rule_based(self, text: str) -> Tuple[str, float]:
        """
        Classify using rule-based patterns.
        
        Args:
            text: Statement text
        
        Returns:
            Tuple of (type, confidence)
        """
        text_lower = text.lower()
        scores = {}
        
        for stype, patterns in self.TYPE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            
            # Normalize by pattern count
            if len(patterns) > 0:
                scores[stype] = score / len(patterns)
            else:
                scores[stype] = 0
        
        # Get best match
        if not scores or max(scores.values()) == 0:
            return ("Assertion/Fact", 0.5)  # Default
        
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type], 1.0)
        
        # Boost confidence if multiple patterns match
        if confidence > 0.3:
            confidence = min(confidence * 1.2, 1.0)
        
        return (best_type, confidence)
    
    def classify_ml(self, text: str) -> Tuple[str, float]:
        """
        Classify using ML model.
        
        Args:
            text: Statement text
        
        Returns:
            Tuple of (type, confidence)
        """
        if not self.ml_model or not self.ml_tokenizer:
            return self.classify_rule_based(text)
        
        try:
            inputs = self.ml_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.ml_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_idx].item()
            
            predicted_type = self.TYPES[predicted_idx]
            return (predicted_type, confidence)
        
        except Exception as e:
            print(f"ML classification error: {e}")
            return self.classify_rule_based(text)
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Hybrid classification: rule-based first, ML as tie-breaker.
        
        Args:
            text: Statement text
        
        Returns:
            Tuple of (type, confidence)
        """
        # Rule-based classification
        rule_type, rule_confidence = self.classify_rule_based(text)
        
        # If rule-based confidence is high, use it
        if rule_confidence >= 0.7:
            return (rule_type, rule_confidence)
        
        # Otherwise, use ML if available
        if self.use_ml:
            ml_type, ml_confidence = self.classify_ml(text)
            # Use ML if confidence is higher
            if ml_confidence > rule_confidence:
                return (ml_type, ml_confidence)
        
        return (rule_type, rule_confidence)
    
    def type_statements(self, statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add type classification to statements.
        
        Args:
            statements: List of Core Statement dictionaries
        
        Returns:
            Statements with added 'type' and 'type_confidence' fields
        """
        typed_statements = []
        
        for statement in statements:
            text = statement.get('text', '')
            stype, confidence = self.classify(text)
            
            statement['type'] = stype
            statement['type_confidence'] = confidence
            typed_statements.append(statement)
        
        return typed_statements


def type_statements(
    statements: List[Dict[str, Any]], 
    use_ml: bool = False,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for statement typing.
    
    Args:
        statements: List of Core Statement dictionaries
        use_ml: Whether to use ML classifier
        model_name: Optional transformer model name
    
    Returns:
        Typed statements
    """
    typer = StatementTyper(use_ml=use_ml, model_name=model_name)
    return typer.type_statements(statements)



