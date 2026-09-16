def evaluate_news2_risk(spo2: float, heart_rate: float, systolic_bp: float, resp_rate: float) -> tuple[int, str]:
    """Evaluates clinical parameters against standard NEWS2 thresholds."""
    score = 0
    
    # SpO2
    if spo2 <= 91: score += 3
    elif spo2 in [92, 93]: score += 2
    elif spo2 in [94, 95]: score += 1

    # Heart Rate
    if heart_rate <= 40 or heart_rate >= 131: score += 3
    elif heart_rate >= 111: score += 2
    elif heart_rate <= 50 or heart_rate >= 91: score += 1

    # Respiratory Rate
    if resp_rate <= 8 or resp_rate >= 25: score += 3
    elif resp_rate >= 21: score += 2

    alert_status = "RED ALERT (DYNAMIC REROUTE)" if (score >= 5 or spo2 < 90) else "GREEN NORMAL (PRIMARY ROUTE)"
    return score, alert_status