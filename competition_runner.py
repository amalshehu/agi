"""
ARC Prize 2025 Competition Runner
Official competition pipeline with proper test set handling
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


from data_loader import load_official_arc_data, ARCTask
from model_loader import load_best_trained_agi
from kaggle_competition_notebook import ARCCompetitionSystem


class OfficialCompetitionRunner:
    """Handles the official ARC Prize 2025 competition workflow"""
    
    def __init__(self, use_trained_model: bool = True):
        self.agi_system = ARCCompetitionSystem()
        
        # Try to load trained model
        if use_trained_model:
            try:
                trained_agi = load_best_trained_agi()
                self.agi_system.agi = trained_agi
                print("✅ Using trained AGI model")
            except Exception as e:
                print(f"⚠️ Could not load trained model: {e}")
                print("Using untrained AGI model")
        
    async def run_evaluation_phase(self, limit: Optional[int] = None) -> str:
        """Run evaluation phase with known solutions for validation"""
        print("🧪 Running Evaluation Phase (with known solutions)")
        
        train_set, eval_set, _ = load_official_arc_data()
        
        # Analyze training data
        self.agi_system.analyze_training_data(train_set, limit=200)
        
        # Process evaluation tasks (these have known solutions)
        eval_tasks = eval_set[:limit] if limit else eval_set
        print(f"Processing {len(eval_tasks)} evaluation tasks...")
        
        submission_data = {}
        correct_count = 0
        total_count = 0
        
        for i, task in enumerate(eval_tasks):
            print(f"Processing eval task {i+1}/{len(eval_tasks)}: {task.id}")
            
            result = await self.agi_system.process_task_with_agi(task)
            
            # Format predictions
            task_predictions = []
            for j, pred in enumerate(result['predictions']):
                task_predictions.append({
                    "attempt_1": pred['attempt_1'].tolist(),
                    "attempt_2": pred['attempt_2'].tolist()
                })
                
                # Check accuracy if we have the solution
                if j < len(task.test) and task.test[j][1] is not None:
                    expected = task.test[j][1]
                    attempt1_correct = (pred['attempt_1'].shape == expected.shape and 
                                     (pred['attempt_1'] == expected).all())
                    attempt2_correct = (pred['attempt_2'].shape == expected.shape and 
                                     (pred['attempt_2'] == expected).all())
                    
                    if attempt1_correct or attempt2_correct:
                        correct_count += 1
                    total_count += 1
            
            submission_data[task.id] = task_predictions
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_file = f"evaluation_results_{timestamp}.json"
        
        with open(eval_file, 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"✅ Evaluation Phase Complete!")
        print(f"📊 Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
        print(f"📄 Results saved: {eval_file}")
        
        return eval_file
    
    async def run_test_phase(self) -> str:
        """Run test phase for final submission (no known solutions)"""
        print("🚀 Running Test Phase (for final submission)")
        
        train_set, eval_set, test_set = load_official_arc_data()
        
        # ONLY use training data (NEVER evaluation data per competition rules)
        self.agi_system.analyze_training_data(train_set, limit=500)
        
        print(f"Processing {len(test_set)} test tasks for final submission...")
        
        submission_data = {}
        
        for i, task in enumerate(test_set):
            print(f"Processing test task {i+1}/{len(test_set)}: {task.id}")
            
            result = await self.agi_system.process_task_with_agi(task)
            
            # Format for submission
            task_predictions = []
            for pred in result['predictions']:
                task_predictions.append({
                    "attempt_1": pred['attempt_1'].tolist(),
                    "attempt_2": pred['attempt_2'].tolist()
                })
            
            submission_data[task.id] = task_predictions
        
        # Save final submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f"final_submission_{timestamp}.json"
        
        with open(submission_file, 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        print(f"🏆 Final Submission Ready!")
        print(f"📄 Submission file: {submission_file}")
        print(f"📊 Processed {len(submission_data)} test tasks")
        
        return submission_file
    
    async def run_full_pipeline(self, eval_limit: Optional[int] = None) -> Dict[str, str]:
        """Run both evaluation and test phases"""
        print("🎯 Starting Full Competition Pipeline")
        
        # Phase 1: Evaluation (for validation)
        eval_file = await self.run_evaluation_phase(limit=eval_limit)
        
        # Phase 2: Test (for final submission) 
        submission_file = await self.run_test_phase()
        
        return {
            'evaluation_results': eval_file,
            'final_submission': submission_file
        }
    
    def validate_submission(self, submission_file: str, is_test_submission: bool = False) -> bool:
        """Validate submission format"""
        try:
            with open(submission_file, 'r') as f:
                data = json.load(f)
            
            print(f"🔍 Validating submission: {submission_file}")
            
            # Load appropriate set to check task IDs
            train_set, eval_set, test_set = load_official_arc_data()
            if is_test_submission:
                expected_tasks = {task.id for task in test_set}
                print("Validating against test set task IDs")
            else:
                expected_tasks = {task.id for task in eval_set}
                print("Validating against evaluation set task IDs")
            
            # Check all required tasks are present
            missing_tasks = expected_tasks - set(data.keys())
            extra_tasks = set(data.keys()) - expected_tasks
            
            if missing_tasks:
                print(f"❌ Missing tasks: {missing_tasks}")
                return False
            
            if extra_tasks:
                print(f"⚠️ Extra tasks (will be ignored): {extra_tasks}")
            
            # Validate format
            for task_id, predictions in data.items():
                if not isinstance(predictions, list):
                    print(f"❌ Task {task_id}: predictions should be a list")
                    return False
                
                for i, pred in enumerate(predictions):
                    if not isinstance(pred, dict):
                        print(f"❌ Task {task_id}[{i}]: should be a dict")
                        return False
                    
                    if "attempt_1" not in pred or "attempt_2" not in pred:
                        print(f"❌ Task {task_id}[{i}]: missing attempts")
                        return False
            
            print(f"✅ Submission format is valid!")
            print(f"📊 Tasks: {len(data)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False


async def main():
    """Main competition runner"""
    runner = OfficialCompetitionRunner()
    
    # For testing, run evaluation phase only
    print("🧪 Running evaluation phase for testing...")
    eval_results = await runner.run_evaluation_phase()
    
    # Validate the result (evaluation results, not test submission)
    runner.validate_submission(eval_results, is_test_submission=False)
    
    print("\n🎯 To run full competition:")
    print("python -c \"import asyncio; from competition_runner import OfficialCompetitionRunner; asyncio.run(OfficialCompetitionRunner().run_full_pipeline())\"")


if __name__ == "__main__":
    asyncio.run(main())
