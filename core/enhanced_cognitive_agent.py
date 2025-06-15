"""
Enhanced CognitiveAgent that incorporates winning strategies from ARC Prize 2024.
Uses deep learning-guided program synthesis, test-time training, and ensemble methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add arc directory to path to import advanced solver
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'arc'))

try:
    from advanced_arc_solver import solve_with_advanced_methods
    from arc_prize_solvers import solve_non_uniform_improved, solve_identity, solve_uniform_mapping
    ADVANCED_SOLVER_AVAILABLE = True
except ImportError:
    ADVANCED_SOLVER_AVAILABLE = False
    print("Warning: Advanced solver not available, falling back to basic methods")

class EnhancedCognitiveAgent:
    """
    Enhanced cognitive agent for ARC puzzle solving using state-of-the-art techniques.
    """
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or "enhanced-cognitive-agent"
        self._challenge = None
        self._last_prediction = None
        self._last_reward = None
        self._solving_history = []
        self._learned_patterns = {}
        
    def load_challenge(self, challenge: Dict):
        """Load an ARC challenge for solving."""
        self._challenge = challenge
        self._last_prediction = None
    
    def run_episode(self):
        """Solve the loaded challenge using advanced techniques."""
        if self._challenge is None:
            raise RuntimeError("Call load_challenge() first.")
        
        # Record solving attempt
        attempt_info = {
            'challenge_id': self._challenge.get('id', 'unknown'),
            'input_shape': None,
            'output_shape': None,
            'method_used': None,
            'success': False
        }
        
        try:
            # Get basic shape information
            if self._challenge.get('train') and len(self._challenge['train']) > 0:
                first_train = self._challenge['train'][0]
                attempt_info['input_shape'] = np.array(first_train['input']).shape
                if 'output' in first_train:
                    attempt_info['output_shape'] = np.array(first_train['output']).shape
            
            # Try advanced solver first
            if ADVANCED_SOLVER_AVAILABLE:
                try:
                    result = solve_with_advanced_methods(self._challenge, debug=False)
                    if result is not None and result.size > 0:
                        self._last_prediction = result
                        attempt_info['method_used'] = 'advanced_ensemble'
                        attempt_info['success'] = True
                        self._solving_history.append(attempt_info)
                        return
                except Exception as e:
                    print(f"Advanced solver failed: {e}")
            
            # Fallback to rule-based approach with improvements
            result = self._solve_with_rule_based_approach()
            if result is not None:
                self._last_prediction = result
                attempt_info['method_used'] = 'rule_based_improved'
                attempt_info['success'] = True
            else:
                # Last resort: return identity
                inp = np.array(self._challenge['train'][0]['input'], dtype=int)
                self._last_prediction = inp.copy()
                attempt_info['method_used'] = 'identity_fallback'
        
        except Exception as e:
            print(f"Error in run_episode: {e}")
            # Emergency fallback
            try:
                inp = np.array(self._challenge['train'][0]['input'], dtype=int)
                self._last_prediction = inp.copy()
                attempt_info['method_used'] = 'error_fallback'
            except:
                self._last_prediction = np.array([[0]], dtype=int)
                attempt_info['method_used'] = 'zero_fallback'
        
        self._solving_history.append(attempt_info)
    
    def _solve_with_rule_based_approach(self) -> Optional[np.ndarray]:
        """Fallback rule-based solving with pattern analysis."""
        ch = self._challenge
        
        if not ch.get('train') or len(ch['train']) == 0:
            return None
        
        # Analyze the task structure
        train_examples = []
        for example in ch['train']:
            inp = np.array(example['input'], dtype=int)
            if 'output' in example:
                out = np.array(example['output'], dtype=int)
                train_examples.append((inp, out))
        
        if not train_examples:
            return None
        
        # Get first example for shape analysis
        inp, out = train_examples[0]
        h0, w0 = inp.shape
        h1, w1 = out.shape
        
        # Enhanced categorization and solving
        if (h0, w0) == (h1, w1):
            # Same shape - try multiple approaches
            
            # First try advanced pattern detection
            if len(train_examples) > 1:
                if self._try_multi_example_patterns(train_examples):
                    return self._last_prediction
            
            # Try identity solver
            try:
                result = solve_identity(ch, ch)
                if result is not None and result.shape == out.shape:
                    return result
            except:
                pass
            
            # Try improved non-uniform solver
            try:
                result = solve_non_uniform_improved(ch, ch, debug=False)
                if result is not None:
                    return result
            except:
                pass
        
        elif h0 > 0 and w0 > 0 and h1 % h0 == 0 and w1 % w0 == 0:
            # Potential uniform scaling
            try:
                result = solve_uniform_mapping(ch, ch)
                if result is not None:
                    return result
            except:
                pass
        
        else:
            # Non-uniform transformation
            try:
                result = solve_non_uniform_improved(ch, ch, debug=False)
                if result is not None:
                    return result
            except:
                pass
        
        return None
    
    def _try_multi_example_patterns(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Try to find patterns across multiple training examples."""
        if len(train_examples) < 2:
            return False
        
        # Check for consistent transformations
        transformations = [
            ('identity', lambda x: x),
            ('flip_h', lambda x: np.fliplr(x)),
            ('flip_v', lambda x: np.flipud(x)),
            ('rot_90', lambda x: np.rot90(x, 1)),
            ('rot_180', lambda x: np.rot90(x, 2)),
            ('rot_270', lambda x: np.rot90(x, 3)),
        ]
        
        # Test each transformation on all examples
        for name, transform in transformations:
            consistent = True
            for inp, out in train_examples:
                try:
                    transformed = transform(inp)
                    if not np.array_equal(transformed, out):
                        # Try with color mapping
                        color_map = self._learn_color_mapping(transformed, out)
                        if color_map:
                            mapped = self._apply_color_mapping(transformed, color_map)
                            if not np.array_equal(mapped, out):
                                consistent = False
                                break
                        else:
                            consistent = False
                            break
                except:
                    consistent = False
                    break
            
            if consistent:
                # Apply this transformation to the test input
                try:
                    test_input = np.array(self._challenge['test'][0]['input'], dtype=int)
                    result = transform(test_input)
                    
                    # Apply color mapping if needed
                    if len(train_examples) > 0:
                        inp, out = train_examples[0]
                        transformed_inp = transform(inp)
                        color_map = self._learn_color_mapping(transformed_inp, out)
                        if color_map:
                            result = self._apply_color_mapping(result, color_map)
                    
                    self._last_prediction = result
                    return True
                except:
                    continue
        
        return False
    
    def _learn_color_mapping(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict[int, int]]:
        """Learn color mapping from input to output."""
        if inp.shape != out.shape:
            return None
        
        mapping = {}
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                old_c, new_c = int(inp[i, j]), int(out[i, j])
                if old_c in mapping and mapping[old_c] != new_c:
                    return None  # Inconsistent mapping
                mapping[old_c] = new_c
        
        return mapping
    
    def _apply_color_mapping(self, grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Apply color mapping to grid."""
        result = grid.copy()
        for old_color, new_color in mapping.items():
            result[grid == old_color] = new_color
        return result
    
    def get_output(self) -> List[List[int]]:
        """Return the last prediction as a list of lists."""
        if self._last_prediction is None:
            raise RuntimeError("Call run_episode() first.")
        return self._last_prediction.tolist()
    
    def learn(self, reward: float):
        """Update learning based on reward."""
        self._last_reward = reward
        
        # Update solving history
        if self._solving_history:
            self._solving_history[-1]['reward'] = reward
            
            # Learn from successful attempts
            if reward > 0.5:  # Success threshold
                last_attempt = self._solving_history[-1]
                method = last_attempt.get('method_used')
                if method and method not in self._learned_patterns:
                    self._learned_patterns[method] = {'successes': 0, 'total': 0}
                
                self._learned_patterns[method]['successes'] += 1
                self._learned_patterns[method]['total'] += 1
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        total_attempts = len(self._solving_history)
        successful_attempts = sum(1 for attempt in self._solving_history if attempt.get('reward', 0) > 0.5)
        
        method_stats = {}
        for method, stats in self._learned_patterns.items():
            if stats['total'] > 0:
                method_stats[method] = {
                    'success_rate': stats['successes'] / stats['total'],
                    'total_uses': stats['total']
                }
        
        return {
            'agent_id': self.agent_id,
            'last_reward': self._last_reward,
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0.0,
            'method_statistics': method_stats,
            'advanced_solver_available': ADVANCED_SOLVER_AVAILABLE
        }
    
    async def process_input(self, text: str) -> str:
        """Process text input (for compatibility)."""
        return f"Processed: {text}"

# Alias for backward compatibility
CognitiveAgent = EnhancedCognitiveAgent
