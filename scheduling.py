from datetime import datetime, timedelta
from typing import Dict, List
from data_models import Flashcard, CategoryPerformance


def calculate_category_multiplier(attempts: int, correct: int) -> float:
    if attempts == 0:
        return 1.0  # Default multiplier for new categories
    
    accuracy = (correct / attempts) * 100
    
    if accuracy < 70:
        return 0.6
    elif accuracy <= 85:
        return 1.0
    else:
        return 1.5


def update_performance(card: Flashcard, is_correct: bool, performance_data: Dict[str, CategoryPerformance]) -> Dict[str, CategoryPerformance]:
    # Update the card's own statistics
    card.attempts += 1
    if is_correct:
        card.correct += 1
    card.lastReview = datetime.now().isoformat()
    
    # Update statistics for each category this card belongs to
    for category_path in card.categories:
        # Create category performance object if it doesn't exist
        if category_path not in performance_data:
            performance_data[category_path] = CategoryPerformance(
                category_path=category_path,
                attempts=0,
                correct=0,
                accuracy=0.0,
                multiplier=1.0
            )
        
        # Update attempts and correct count
        perf = performance_data[category_path]
        perf.attempts += 1
        if is_correct:
            perf.correct += 1
        
        # Recalculate accuracy and multiplier
        perf.accuracy = (perf.correct / perf.attempts) * 100 if perf.attempts > 0 else 0.0
        perf.multiplier = calculate_category_multiplier(perf.attempts, perf.correct)
    
    return performance_data


def calculate_next_review(card: Flashcard, is_correct: bool, performance_data: Dict[str, CategoryPerformance]) -> datetime:
    # Step 1: Determine base interval
    base_interval_days = 3 if is_correct else 1
    
    # Step 2 & 3: Get all category multipliers and calculate combined multiplier
    category_multipliers = []
    for category_path in card.categories:
        if category_path in performance_data:
            multiplier = performance_data[category_path].multiplier
            category_multipliers.append(multiplier)
        else:
            # If category doesn't exist yet (shouldn't happen after update_performance)
            category_multipliers.append(1.0)
    
    # Calculate average multiplier (combined multiplier)
    combined_multiplier = sum(category_multipliers) / len(category_multipliers) if category_multipliers else 1.0
    
    # Step 4: Calculate final interval
    final_interval_days = base_interval_days * combined_multiplier
    
    # Step 5: Calculate next review time
    current_time = datetime.now()
    next_review = current_time + timedelta(days=final_interval_days)
    
    return next_review


def get_category_statistics(performance_data: Dict[str, CategoryPerformance]) -> Dict:
    if not performance_data:
        return {
            "total_categories": 0,
            "struggling_categories": 0,
            "moderate_categories": 0,
            "strong_categories": 0,
            "overall_accuracy": 0.0,
            "total_attempts": 0
        }
    
    total_attempts = 0
    total_correct = 0
    struggling = 0
    moderate = 0
    strong = 0
    
    for perf in performance_data.values():
        total_attempts += perf.attempts
        total_correct += perf.correct
        
        accuracy = perf.accuracy
        if accuracy < 70:
            struggling += 1
        elif accuracy <= 85:
            moderate += 1
        else:
            strong += 1
    
    overall_accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0.0
    
    return {
        "total_categories": len(performance_data),
        "struggling_categories": struggling,
        "moderate_categories": moderate,
        "strong_categories": strong,
        "overall_accuracy": round(overall_accuracy, 1),
        "total_attempts": total_attempts
    }


def process_review_result(card: Flashcard, is_correct: bool, performance_data: Dict[str, CategoryPerformance]) -> tuple[Flashcard, Dict[str, CategoryPerformance]]:
    # Update all performance statistics
    performance_data = update_performance(card, is_correct, performance_data)
    
    # Calculate and set next review time
    next_review = calculate_next_review(card, is_correct, performance_data)
    card.nextReview = next_review.isoformat()
    
    return card, performance_data
