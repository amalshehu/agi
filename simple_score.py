"""
Simple scoring for ARC submission
"""
import json
import numpy as np
from data_loader import load_official_arc_data

def score_submission(submission_file: str):
    """Score submission against evaluation set"""
    print(f"📊 Scoring: {submission_file}")
    
    # Load our official data
    train_set, eval_set, test_set = load_official_arc_data()
    
    # Load submission
    with open(submission_file, 'r') as f:
        submission = json.load(f)
    
    print(f"Submission: {len(submission)} tasks")
    print(f"Eval set: {len(eval_set)} tasks")
    
    # Score against evaluation set
    correct_predictions = 0
    total_predictions = 0
    task_scores = []
    
    for task in eval_set:
        if task.id in submission:
            task_correct = 0
            task_total = 0
            
            predictions = submission[task.id]
            
            for i, (test_input, test_output) in enumerate(task.test):
                if test_output is not None and i < len(predictions):
                    pred = predictions[i]
                    
                    # Check both attempts
                    attempt1 = np.array(pred["attempt_1"])
                    attempt2 = np.array(pred["attempt_2"])
                    
                    correct = False
                    if (attempt1.shape == test_output.shape and 
                        np.array_equal(attempt1, test_output)):
                        correct = True
                    elif (attempt2.shape == test_output.shape and 
                          np.array_equal(attempt2, test_output)):
                        correct = True
                    
                    if correct:
                        correct_predictions += 1
                        task_correct += 1
                    
                    total_predictions += 1
                    task_total += 1
            
            if task_total > 0:
                task_score = task_correct / task_total
                task_scores.append(task_score)
                if task_score > 0:
                    print(f"✅ Task {task.id}: {task_score:.2f} ({task_correct}/{task_total})")
    
    # Overall score
    if total_predictions > 0:
        overall_score = correct_predictions / total_predictions
        print(f"\n🎯 Overall Score: {overall_score:.3f} ({correct_predictions}/{total_predictions})")
    else:
        print("❌ No valid predictions found")
    
    # Task-level stats
    if task_scores:
        solved_tasks = sum(1 for s in task_scores if s == 1.0)
        print(f"📊 Perfectly solved tasks: {solved_tasks}/{len(task_scores)}")
        print(f"📊 Average task score: {np.mean(task_scores):.3f}")
    
    return overall_score if total_predictions > 0 else 0.0

if __name__ == "__main__":
    import sys
    import glob
    from pathlib import Path
    
    if len(sys.argv) > 1:
        # Score specific file
        score = score_submission(sys.argv[1])
    else:
        # Score evaluation results (these have the right task IDs)
        eval_results = glob.glob("evaluation_results*.json")
        if eval_results:
            latest = max(eval_results, key=lambda x: Path(x).stat().st_mtime)
            print(f"\n📋 Scoring evaluation results: {latest}")
            score = score_submission(latest)
        else:
            print("❌ No evaluation results found")
            print("Available files:")
            for f in glob.glob("*.json"):
                print(f"  {f}")
