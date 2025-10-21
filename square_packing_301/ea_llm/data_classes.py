from dataclasses import dataclass
from typing import Optional, Dict, Any
from config import NUM_SQUARES

@dataclass
class Program:
    """Data class representing a square packing program with evaluation metrics"""
    id: str
    code: str
    num_squares: int = NUM_SQUARES
    overlaps: int = 0
    is_valid: bool = False
    side_length: float = 0.0
    efficiency: float = 0.0
    llm_evaluation: Optional[Dict[str, Any]] = None
    fitness_score: Optional[float] = None
    solution_path: Optional[str] = None
    
    @property
    def fitness(self) -> float:
        """Calculate fitness score based on efficiency and validity"""
        if self.fitness_score is not None:
            return self.fitness_score
        
        if not self.is_valid:
            return 0.0
        
        # Base fitness on efficiency, penalize overlaps
        fitness = self.efficiency
        if self.overlaps > 0:
            fitness -= (self.overlaps * 10)  # Penalty for overlaps
        
        return max(0.0, fitness)
    
    def update_evaluation(self, eval_results: Dict[str, Any]) -> None:
        """Update program evaluation from results dictionary"""
        self.num_squares = eval_results.get('num_squares', self.num_squares)
        self.overlaps = eval_results.get('overlaps', self.overlaps)
        self.is_valid = eval_results.get('is_valid', self.is_valid)
        self.side_length = eval_results.get('side_length', self.side_length)
        self.efficiency = eval_results.get('efficiency', self.efficiency)
        self.llm_evaluation = eval_results.get('llm_evaluation', self.llm_evaluation)
        
        # Recalculate fitness score if needed
        self.fitness_score = None