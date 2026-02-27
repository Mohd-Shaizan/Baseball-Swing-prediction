def map_outcome(outcome):
    outcome = outcome.lower()
    
    if "hit" in outcome or "single" in outcome or "double" in outcome or "home" in outcome:
        return "Hit"
    
    elif "swing" in outcome or "foul" in outcome:
        return "Swing"
    
    else:
        return "Strike"