"""
Targeted extraction module for entities, actions, logic, uncertainty, and attack signals.
"""

# Apply Python 3.12 compatibility patch before importing spacy
try:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import fix_pydantic_python312
    except ImportError:
        pass
except Exception:
    pass

import re
import spacy
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict


class Extractor:
    """Extracts policy-relevant patterns from statements."""
    
    # Domain-specific entity patterns
    BIORISK_DOMAINS = [
        'biosecurity', 'biosafety', 'biodefense', 'biological threat',
        'pathogen', 'virus', 'bacteria', 'toxin', 'biological weapon',
        'dual-use', 'gain-of-function', 'pandemic', 'epidemic'
    ]
    
    HAZARD_INDICATORS = [
        'hazard', 'risk', 'danger', 'threat', 'vulnerability',
        'exposure', 'contamination', 'outbreak', 'spill'
    ]
    
    MATERIALS = [
        'pathogen', 'virus', 'bacteria', 'toxin', 'chemical',
        'biological agent', 'sample', 'culture', 'specimen'
    ]
    
    TOOLS = [
        'laboratory', 'lab', 'equipment', 'facility', 'biosafety cabinet',
        'autoclave', 'incubator', 'sequencer', 'PCR'
    ]
    
    DATASETS = [
        'database', 'dataset', 'sequence data', 'genomic data',
        'protein data', 'research data'
    ]
    
    FACILITIES = [
        'laboratory', 'lab', 'facility', 'biosafety level',
        'BSL-1', 'BSL-2', 'BSL-3', 'BSL-4', 'containment'
    ]
    
    ROLES = [
        'technician', 'scientist', 'researcher', 'advisor',
        'consultant', 'expert', 'analyst', 'operator'
    ]
    
    # Action patterns
    ACTION_PATTERNS = {
        'Planning': [
            r'\b(plan|strategy|prepare|organize|design|develop)\b',
            r'\b(intend|aim|goal|objective|purpose)\b',
        ],
        'Bypassing': [
            r'\b(bypass|circumvent|avoid|evade|sidestep)\b',
            r'\b(work around|get around|find a way)\b',
        ],
        'Manipulating': [
            r'\b(manipulate|modify|alter|change|edit)\b',
            r'\b(tamper|adjust|tweak|modify)\b',
        ],
        'Producing': [
            r'\b(produce|create|generate|manufacture|synthesize)\b',
            r'\b(make|build|construct|develop)\b',
        ],
        'Advising': [
            r'\b(advise|recommend|suggest|propose|counsel)\b',
            r'\b(guidance|direction|instruction)\b',
        ],
        'Warning': [
            r'\b(warn|caution|alert|notify|inform)\b',
            r'\b(danger|risk|hazard|threat)\b',
        ],
        'Instructing': [
            r'\b(instruct|direct|command|order|tell)\b',
            r'\b(should|must|need to|required)\b',
        ],
    }
    
    # Uncertainty patterns
    UNCERTAINTY_PATTERNS = [
        r'\b(might|may|could|possibly|perhaps|maybe)\b',
        r'\b(potentially|likely|unlikely|probably|possibly)\b',
        r'\b(uncertain|unclear|ambiguous|unpredictable)\b',
        r'\b(depends|varies|fluctuates|changes)\b',
        r'\b(some|certain|various|several|many)\b',
        r'\b(approximately|roughly|about|around)\b',
    ]
    
    # Attack signal patterns
    ATTACK_PATTERNS = {
        'role_flipping': [
            r'\b(ignore|disregard|forget|pretend)\b.*\b(guidelines|rules|safety|policy)\b',
            r'\b(as|pretend to be|act as|roleplay)\b.*\b(hacker|h4xxor|expert|professional)\b',
        ],
        'prompt_injection': [
            r'\b(ignore previous|forget all|new instructions|system prompt)\b',
            r'\b(override|bypass|circumvent)\b.*\b(safety|filter|guard)\b',
        ],
        'obfuscated_text': [
            r'\b(ROT13|base64|encoded|decoded|cipher)\b',
            r'\b(xor|encrypt|decrypt|obfuscate)\b',
        ],
        'chain_of_thought': [
            r'\b(think step by step|reason through|break down)\b',
            r'\b(consider|imagine|suppose|hypothetically)\b',
        ],
        'policy_circumvention': [
            r'\b(circumvent|bypass|work around|find loophole)\b',
            r'\b(technicality|exception|edge case)\b',
        ],
    }
    
    def __init__(self, nlp_model=None):
        """
        Initialize the extractor.
        
        Args:
            nlp_model: Optional pre-loaded spaCy model
        """
        if nlp_model is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError("spaCy model required for extraction")
        else:
            self.nlp = nlp_model
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract domain entities from text.
        
        Args:
            text: Statement text
        
        Returns:
            Dictionary of entity types to lists of entities
        """
        text_lower = text.lower()
        entities = {
            'biorisk_domains': [],
            'hazard_indicators': [],
            'materials': [],
            'tools': [],
            'datasets': [],
            'facilities': [],
            'roles': []
        }
        
        # Extract using patterns
        all_patterns = {
            'biorisk_domains': self.BIORISK_DOMAINS,
            'hazard_indicators': self.HAZARD_INDICATORS,
            'materials': self.MATERIALS,
            'tools': self.TOOLS,
            'datasets': self.DATASETS,
            'facilities': self.FACILITIES,
            'roles': self.ROLES,
        }
        
        for entity_type, patterns in all_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', text_lower):
                    if pattern not in entities[entity_type]:
                        entities[entity_type].append(pattern)
        
        # Also use spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                # Could add to entities if relevant
                pass
        
        return entities
    
    def extract_actions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract action frames from text.
        
        Args:
            text: Statement text
        
        Returns:
            List of action dictionaries
        """
        actions = []
        text_lower = text.lower()
        
        for action_type, patterns in self.ACTION_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    actions.append({
                        'type': action_type,
                        'text': match.group(0),
                        'span': [match.start(), match.end()],
                        'pattern': pattern
                    })
        
        return actions
    
    def extract_logic(self, text: str) -> Dict[str, Any]:
        """
        Extract logical structure (if-then, causal links, constraints).
        
        Args:
            text: Statement text
        
        Returns:
            Dictionary with logic structure
        """
        logic = {
            'condition': None,
            'consequence': None,
            'causal_links': [],
            'constraints': []
        }
        
        # Extract if-then patterns
        if_then_pattern = r'\b(if|when|unless|provided that)\s+([^,]+?)\s*,\s*(then|so|therefore)?\s*([^.]+)'
        if_match = re.search(if_then_pattern, text, re.IGNORECASE)
        if if_match:
            logic['condition'] = if_match.group(2).strip()
            logic['consequence'] = if_match.group(4).strip() if len(if_match.groups()) > 3 else None
        
        # Extract causal links
        causal_patterns = [
            r'\b(because|since|as|due to|owing to)\s+([^.]+)',
            r'\b(leads to|results in|causes|triggers)\s+([^.]+)',
        ]
        for pattern in causal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                logic['causal_links'].append(match.group(0))
        
        # Extract constraints
        constraint_patterns = [
            r'\b(must|should|required|necessary|mandatory)\s+([^.]+)',
            r'\b(only|solely|exclusively|merely)\s+([^.]+)',
        ]
        for constraint_pattern in constraint_patterns:
            matches = re.finditer(constraint_pattern, text, re.IGNORECASE)
            for match in matches:
                logic['constraints'].append(match.group(0))
        
        return logic
    
    def extract_uncertainty(self, text: str) -> Dict[str, Any]:
        """
        Extract uncertainty indicators.
        
        Args:
            text: Statement text
        
        Returns:
            Dictionary with uncertainty information
        """
        uncertainty = {
            'hedge': False,
            'hedge_tokens': [],
            'modal_verbs': [],
            'probability_cues': [],
            'confidence': 1.0
        }
        
        text_lower = text.lower()
        
        # Extract hedge tokens
        for pattern in self.UNCERTAINTY_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                token = match.group(0)
                if token not in uncertainty['hedge_tokens']:
                    uncertainty['hedge_tokens'].append(token)
        
        if uncertainty['hedge_tokens']:
            uncertainty['hedge'] = True
            # Reduce confidence based on number of hedges
            uncertainty['confidence'] = max(0.5, 1.0 - (len(uncertainty['hedge_tokens']) * 0.1))
        
        # Extract modal verbs
        modal_verbs = ['might', 'may', 'could', 'should', 'would', 'must', 'can']
        for modal in modal_verbs:
            if re.search(r'\b' + modal + r'\b', text_lower):
                uncertainty['modal_verbs'].append(modal)
        
        # Extract probability cues
        probability_patterns = [
            r'\b(likely|unlikely|probably|possibly|certainly|definitely)\b',
            r'\b(probability|chance|odds|likelihood)\b',
        ]
        for pattern in probability_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                uncertainty['probability_cues'].append(match.group(0))
        
        return uncertainty
    
    def extract_attack_signals(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract jailbreak/attack signals.
        
        Args:
            text: Statement text
        
        Returns:
            List of attack signal dictionaries
        """
        attack_signals = []
        text_lower = text.lower()
        
        for signal_type, patterns in self.ATTACK_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    attack_signals.append({
                        'type': signal_type,
                        'text': match.group(0),
                        'span': [match.start(), match.end()],
                        'pattern': pattern,
                        'severity': 'high' if signal_type in ['role_flipping', 'prompt_injection'] else 'medium'
                    })
        
        return attack_signals
    
    def extract_all(self, statements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract all patterns from statements.
        
        Args:
            statements: List of Core Statement dictionaries
        
        Returns:
            Dictionary with risk_targets, action_frames, attack_signal
        """
        risk_targets = []
        action_frames = []
        attack_signals = []
        
        for statement in statements:
            text = statement.get('text', '')
            cs_id = statement.get('id', '')
            
            # Extract entities and create risk targets
            entities = self.extract_entities(text)
            if any(entities.values()):
                risk_target = {
                    'target': text[:100],  # First 100 chars as target description
                    'support_cs': [cs_id],
                    'domain': 'biosecurity' if entities.get('biorisk_domains') else 'general',
                    'risk_factor': 'entity_mention',
                    'entities': entities
                }
                risk_targets.append(risk_target)
            
            # Extract actions
            actions = self.extract_actions(text)
            for action in actions:
                action_frame = {
                    'action_type': action['type'],
                    'text': action['text'],
                    'cs_id': cs_id,
                    'span': action['span']
                }
                action_frames.append(action_frame)
            
            # Extract attack signals
            signals = self.extract_attack_signals(text)
            for signal in signals:
                signal['cs_id'] = cs_id
                attack_signals.append(signal)
        
        return {
            'risk_targets': risk_targets,
            'action_frames': action_frames,
            'attack_signal': attack_signals
        }


def extract_patterns(
    statements: List[Dict[str, Any]], 
    nlp_model=None
) -> Dict[str, Any]:
    """
    Convenience function for pattern extraction.
    
    Args:
        statements: List of Core Statement dictionaries
        nlp_model: Optional pre-loaded spaCy model
    
    Returns:
        Dictionary with extracted patterns
    """
    extractor = Extractor(nlp_model)
    return extractor.extract_all(statements)

