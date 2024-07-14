import tkinter as tk
from tkinter import scrolledtext, END
import random
from predictor import predict_class, get_response

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        
        # Chat history display
        self.history = scrolledtext.ScrolledText(root, width=60, height=20)
        self.history.grid(row=0, column=0, padx=10, pady=10)
        
        # Input box
        self.input_box = tk.Entry(root, width=50)
        self.input_box.grid(row=1, column=0, padx=10, pady=10)
        
        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
        
        # Bind <Return> key to send_message function
        self.root.bind('<Return>', self.send_message_enter)
        
        # Welcome message
        self.add_message("Chatbot", "Welcome! How can I assist you today?")
        
    def send_message(self):
        message = self.input_box.get()
        self.add_message("You", message)
        tag = predict_class(message)
        response = get_response(tag)
        self.add_message("Chatbot", response)
        self.input_box.delete(0, END)
        
    def send_message_enter(self, event):
        self.send_message()
        
    def add_message(self, sender, message):
        self.history.config(state=tk.NORMAL)
        self.history.insert(tk.END, f"{sender}: {message}\n\n")
        self.history.config(state=tk.DISABLED)
        self.history.see(tk.END)
        
    

# Main function to start the application
def main():
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
