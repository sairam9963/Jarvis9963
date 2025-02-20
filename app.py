import gradio as gr

def chat_interface(user_input):
    response = generate_response(user_input)
    save_to_memory(user_input, response)
    return response

gr.Interface(fn=chat_interface, inputs="text", outputs="text").launch()
