import cohere

def generate_insight_from_accuracy(accuracy, api_key):
    co = cohere.Client(api_key)

    prompt = f"""
    A Random Forest model was used to classify social media posts as flood-related or not.
    The model achieved an accuracy of {accuracy:.2%}.

    Please provide a clear and detailed explanation of what this accuracy means in the context of the flood classification task. 
    The explanation should help a public audience understand how well the model performs, specifically in detecting flood-related content from social media. 
    Avoid technical jargon and focus on meaningful interpretation of the accuracy. No need to include practical advice or suggestions.
    """

    response = co.chat(
        model="command-r",  # Use "command-r-plus" if available to you
        message=prompt,
        temperature=0.7,
    )

    return response.text.strip()
