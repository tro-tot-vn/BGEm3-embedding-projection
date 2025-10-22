"""
Weight Calculator for Multi-feature Hard Negatives
Supports Max+Incremental and Capped Linear strategies
"""

import json
from pathlib import Path
from typing import List, Union


class WeightCalculator:
    """Calculate combined weights for multiple feature types"""
    
    def __init__(self, config_path: str = "data/weight-config.json"):
        """
        Initialize weight calculator
        
        Args:
            config_path: Path to weight config JSON file
        """
        # Handle both absolute and relative paths
        if not Path(config_path).is_absolute():
            # Assume relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_path
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract base weights (skip metadata keys starting with _)
        self.base_weights = {
            k: v for k, v in config.items() 
            if not k.startswith("_") and isinstance(v, (int, float))
        }
        
        # Load method config from metadata
        metadata = config.get("_metadata", {})
        self.method = metadata.get("method", "max_incremental")
        self.increment_ratio = metadata.get("increment_ratio", 0.3)
        self.cap = metadata.get("cap", 4.0)
        
        print(f"✅ Loaded weight config: {len(self.base_weights)} features")
        print(f"   Method: {self.method}, Ratio: {self.increment_ratio}")
    
    def calculate(self, feature_types: Union[List[str], str]) -> float:
        """
        Calculate combined weight for given feature type(s)
        
        Args:
            feature_types: Single feature type string or list of types
            
        Returns:
            Combined weight as float
            
        Examples:
            >>> calc = WeightCalculator()
            >>> calc.calculate("location")
            2.5
            >>> calc.calculate(["location", "price"])
            3.1
        """
        # Handle single string input
        if isinstance(feature_types, str):
            feature_types = [feature_types]
        
        # Handle empty input
        if not feature_types:
            return 1.0
        
        # Get base weights for each feature
        weights = [
            self.base_weights.get(t, self.base_weights.get("other", 1.0))
            for t in feature_types
        ]
        
        # Calculate based on selected method
        if self.method == "max_incremental":
            return self._max_incremental(weights)
        elif self.method == "capped_linear":
            return self._capped_linear(weights)
        elif self.method == "average":
            return sum(weights) / len(weights)
        elif self.method == "sum":
            return sum(weights)
        else:
            # Default to max_incremental
            return self._max_incremental(weights)
    
    def _max_incremental(self, weights: List[float]) -> float:
        """
        Max + Incremental strategy
        
        Takes the maximum weight (dominant error) and adds
        a percentage of remaining weights as cumulative penalty.
        
        Formula: max(weights) + sum(others) * increment_ratio
        
        Args:
            weights: List of base weights
            
        Returns:
            Combined weight
        """
        if not weights:
            return 1.0
        
        max_weight = max(weights)
        remaining = [w for w in weights if w != max_weight]
        bonus = sum(w * self.increment_ratio for w in remaining)
        
        return max_weight + bonus
    
    def _capped_linear(self, weights: List[float]) -> float:
        """
        Capped Linear strategy
        
        Simply sums all weights but caps at maximum value.
        
        Formula: min(sum(weights), cap)
        
        Args:
            weights: List of base weights
            
        Returns:
            Combined weight (capped)
        """
        if not weights:
            return 1.0
        
        return min(sum(weights), self.cap)
    
    def set_method(self, method: str, **kwargs):
        """
        Change calculation method and parameters
        
        Args:
            method: One of 'max_incremental', 'capped_linear', 'average', 'sum'
            **kwargs: Method-specific parameters
                - increment_ratio: for max_incremental (default: 0.3)
                - cap: for capped_linear (default: 4.0)
        """
        self.method = method
        
        if method == "max_incremental":
            self.increment_ratio = kwargs.get("increment_ratio", 0.3)
        elif method == "capped_linear":
            self.cap = kwargs.get("cap", 4.0)
        
        print(f"✅ Updated method to: {method}")


def main():
    """Test weight calculator"""
    print("=" * 60)
    print("Weight Calculator Test")
    print("=" * 60)
    
    # Initialize
    calc = WeightCalculator()
    
    # Test cases
    test_cases = [
        (["location"], "Single: location"),
        (["price"], "Single: price"),
        (["amenity"], "Single: amenity"),
        (["location", "price"], "Double: location + price"),
        (["location", "amenity"], "Double: location + amenity"),
        (["price", "amenity"], "Double: price + amenity"),
        (["location", "price", "amenity"], "Triple: all important"),
        (["location", "price", "area", "amenity"], "Quad: everything"),
        (["furniture", "floor"], "Low priority features"),
    ]
    
    print("\n📊 Max + Incremental (ratio=0.3):")
    print("-" * 60)
    for types, desc in test_cases:
        weight = calc.calculate(types)
        types_str = ", ".join(types)
        print(f"{desc:35} → {weight:.2f}  ({types_str})")
    
    # Test with different method
    print("\n📊 Capped Linear (cap=4.0):")
    print("-" * 60)
    calc.set_method("capped_linear", cap=4.0)
    for types, desc in test_cases:
        weight = calc.calculate(types)
        types_str = ", ".join(types)
        print(f"{desc:35} → {weight:.2f}  ({types_str})")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
