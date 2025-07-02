def evaluate(model, test_data):
    correct = 0
    total = len(test_data)
    for item in test_data:
        result = model.predict({"question": item["input"]})
        if result["answer"].strip().lower() == item["expected"].strip().lower():
            correct += 1
    return {"accuracy": correct / total if total > 0 else 0}
