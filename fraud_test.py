import requests

url = 'http://127.0.0.1:5000/predict'

data_1 = {
        "age": 41,
        "incident_severity": "Total Loss",
        "total_claim_amount": 71600,
        "insured_hobbies": "chess",
        "policy_state": "OH",
        "number_of_vehicles_involved": 1,
        "property_damage": "YES",
        "auto_model": "92x",
        "insured_occupation": "craft-repair",
        "vehicle_claim": 52080,
        "bodily_injuries": 1,
        "months_as_customer": 328,
        "insured_relationship": "husband",
        "injury_claim": 6510,
        "insured_zip": 466132,
        "witnesses": 2,
        "capital-loss": 0,
        "authorities_contacted": "Police",
        "property_claim": 13020,
        "capital-gains": 53300,
        "incident_type": "Single Vehicle Collision",
        "insured_education_level": "MD",
        "collision_type": "Side Collision",
        "umbrella_limit": 0,
        "policy_number": 521585,
        "policy_csl": "250/500",
        "insured_sex": "MALE",
        "auto_year": 2004,
        "auto_make": "Saab",
        "policy_annual_premium": 1406.91,
        "police_report_available": "YES",
        "incident_hour_of_the_day": 5,
        "policy_deductable": 1000
}

requests.post(url, json = data_1).json()