CORRECTED_CTR = {
    # Perfect match
    "perfect": 0.20,  # Baseline: 20% CTR
    
    # ========================================
    # SINGLE ERRORS (sorted by importance)
    # ========================================
    "wrong_location": 0.05,   # Location critical: 75% decline
    "wrong_price": 0.08,      # Price very important: 60% decline  
    "wrong_area": 0.10,       # Area important: 50% decline
    "wrong_amenity": 0.12,    # Amenity moderate: 40% decline
    
    # ========================================
    # TWO ERRORS (multiplicative model)
    # ========================================
    # Formula: CTR_2 = CTR_1 × CTR_other / perfect
    # Logic: Independent probabilities multiply
    
    "location_price": 0.020,    # 2.0% (0.05 * 0.08 / 0.20)
    "location_area": 0.025,     # 2.5% (0.05 * 0.10 / 0.20)
    "location_amenity": 0.030,  # 3.0% (0.05 * 0.12 / 0.20)
    
    "price_area": 0.040,        # 4.0% (0.08 * 0.10 / 0.20)
    "price_amenity": 0.048,     # 4.8% (0.08 * 0.12 / 0.20)
    
    "area_amenity": 0.060,      # 6.0% (0.10 * 0.12 / 0.20)
    
    # ========================================
    # THREE ERRORS
    # ========================================
    "location_price_area": 0.010,      # (0.05 * 0.08 * 0.10 / (0.20**2))
    "location_price_amenity": 0.012,   # (0.05 * 0.08 * 0.12 / (0.20**2))
    "location_area_amenity": 0.015,    # (0.05 * 0.10 * 0.12 / (0.20**2))
    "price_area_amenity": 0.024,       # (0.08 * 0.10 * 0.12 / (0.20**2))
    
    # ========================================
    # FOUR ERRORS
    # ========================================
    "all_wrong": 0.006  # (0.05 * 0.08 * 0.10 * 0.12 / (0.20**3))
}