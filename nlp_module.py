from transformers import pipeline

qa_pipeline = pipeline("question-answering")

def get_answer(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error in NLP module: {str(e)}"
