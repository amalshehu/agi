"""
Official ARC Prize 2025 Data Loader
Loads the official competition dataset
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ARCTask:
    """Official ARC task structure"""
    id: str
    train: List[Tuple[np.ndarray, np.ndarray]]
    test: List[Tuple[np.ndarray, Optional[np.ndarray]]]


class OfficialARCLoader:
    """Loads official ARC Prize 2025 data"""
    
    def __init__(self, data_dir: str = "arc-prize-2025"):
        self.data_dir = Path(data_dir)
        
    def load_data(self) -> Tuple[List[ARCTask], List[ARCTask]]:
        """Load training and evaluation datasets"""
        
        # Load training data
        train_challenges = self._load_json("arc-agi_training_challenges.json")
        train_solutions = self._load_json("arc-agi_training_solutions.json")
        
        # Load evaluation data
        eval_challenges = self._load_json("arc-agi_evaluation_challenges.json")
        eval_solutions = self._load_json("arc-agi_evaluation_solutions.json")
        
        # Load test data (for submission)
        test_challenges = self._load_json("arc-agi_test_challenges.json")
        
        # Convert to ARCTask objects
        train_tasks = self._convert_to_tasks(train_challenges, train_solutions)
        eval_tasks = self._convert_to_tasks(eval_challenges, eval_solutions)
        test_tasks = self._convert_to_tasks(test_challenges, {})  # No solutions for test
        
        print(f"Loaded {len(train_tasks)} training tasks")
        print(f"Loaded {len(eval_tasks)} evaluation tasks")
        print(f"Loaded {len(test_tasks)} test tasks")
        
        return train_tasks, eval_tasks, test_tasks
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from data directory"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find {file_path}")
            
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _convert_to_tasks(self, challenges: Dict, solutions: Dict) -> List[ARCTask]:
        """Convert JSON data to ARCTask objects"""
        tasks = []
        
        for task_id, challenge_data in challenges.items():
            # Convert training examples
            train_examples = []
            for example in challenge_data['train']:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                train_examples.append((input_grid, output_grid))
            
            # Convert test examples
            test_examples = []
            for example in challenge_data['test']:
                input_grid = np.array(example['input'])
                
                # Get solution if available
                output_grid = None
                if task_id in solutions:
                    if len(solutions[task_id]) > len(test_examples):
                        output_grid = np.array(solutions[task_id][len(test_examples)])
                
                test_examples.append((input_grid, output_grid))
            
            task = ARCTask(
                id=task_id,
                train=train_examples,
                test=test_examples
            )
            tasks.append(task)
        
        return tasks
    
    def get_sample_submission_format(self) -> Dict:
        """Get the sample submission format"""
        sample_path = self.data_dir / "sample_submission.json"
        if sample_path.exists():
            with open(sample_path, 'r') as f:
                return json.load(f)
        return {}


def load_official_arc_data(data_dir: str = "arc-prize-2025") -> Tuple[List[ARCTask], List[ARCTask], List[ARCTask]]:
    """Convenience function to load official ARC data"""
    loader = OfficialARCLoader(data_dir)
    return loader.load_data()


if __name__ == "__main__":
    # Test the loader
    try:
        train_tasks, eval_tasks, test_tasks = load_official_arc_data()
        
        print("\nDataset Summary:")
        print(f"Training: {len(train_tasks)} tasks")
        print(f"Evaluation: {len(eval_tasks)} tasks") 
        print(f"Test: {len(test_tasks)} tasks")
        
        # Show first task details
        if train_tasks:
            first_task = train_tasks[0]
            print(f"\nFirst training task: {first_task.id}")
            print(f"Training examples: {len(first_task.train)}")
            print(f"Test cases: {len(first_task.test)}")
            
            if first_task.train:
                input_shape = first_task.train[0][0].shape
                output_shape = first_task.train[0][1].shape
                print(f"Example input shape: {input_shape}")
                print(f"Example output shape: {output_shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the arc-prize-2025 directory contains the official data files")
