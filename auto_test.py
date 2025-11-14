import pandas as pd
import json
import requests
import time

# Load data
data = pd.read_csv('assets/test_template.csv')

# Load API
api_url = 'http://localhost:8000/query'

# Test queries
for index, row in data.iterrows():
    query = row['question']
    expected_answer = row['expected_answer']
    print(f"Testing query: {query}")
    print("\n")
    print(f"Expected answer: {expected_answer}")
    print("\n")
    
    # Send request
    response = requests.post(api_url, json={'query': query})
    print(f"Response: {response.json()}")
    print("\n")
    
    print("================================================")
    
    # Populate actual_answer field in the original data
    data.at[index, 'actual_answer'] = response.json()['answer']
    
    # Delay 10 seconds between questions to prevent rate limit
    if index < len(data) - 1:  # Don't delay after the last question
        print("Waiting 5 seconds before next query...")
        time.sleep(5)
    
# Save results to csv
data.to_csv('assets/test_results.csv', index=False)