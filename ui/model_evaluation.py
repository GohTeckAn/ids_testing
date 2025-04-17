import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tkinter import messagebox

class ModelEvaluationPage(ctk.CTkFrame):
    def __init__(self, parent, model_manager):
        super().__init__(parent)
        self.model_manager = model_manager
        
        # Create main layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Model Selection Frame
        self.model_frame = ctk.CTkFrame(self)
        self.model_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Model Selection Dropdown
        models = self.model_manager.get_available_models()
        self.selected_model = ctk.StringVar(value=models[0] if models else "")
        
        ctk.CTkLabel(self.model_frame, text="Select Model:").pack(pady=5)
        self.model_dropdown = ctk.CTkOptionMenu(
            self.model_frame,
            values=models,
            variable=self.selected_model
        )
        self.model_dropdown.pack(pady=5)
        
        # Metrics Frame
        self.metrics_frame = ctk.CTkFrame(self)
        self.metrics_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Metrics Labels
        self.accuracy_label = ctk.CTkLabel(self.metrics_frame, text="Accuracy: N/A")
        self.accuracy_label.pack(pady=5)
        
        self.precision_label = ctk.CTkLabel(self.metrics_frame, text="Precision: N/A")
        self.precision_label.pack(pady=5)
        
        self.recall_label = ctk.CTkLabel(self.metrics_frame, text="Recall: N/A")
        self.recall_label.pack(pady=5)
        
        self.f1_label = ctk.CTkLabel(self.metrics_frame, text="F1 Score: N/A")
        self.f1_label.pack(pady=5)
        
        # Confusion Matrix Frame
        self.matrix_frame = ctk.CTkFrame(self)
        self.matrix_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # Create figure for confusion matrix
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.matrix_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Evaluation Button
        self.eval_button = ctk.CTkButton(self, text="Evaluate Model", 
                                       command=self.evaluate_model)
        self.eval_button.grid(row=2, column=0, columnspan=2, pady=10)
        
    def evaluate_model(self):
        try:
            # Get selected model
            model_name = self.selected_model.get()
            if not model_name:
                messagebox.showerror("Error", "Please select a model")
                return
            
            # Evaluate model
            y_pred, y_test = self.model_manager.evaluate_model(model_name)
            if y_pred is None or y_test is None:
                messagebox.showerror("Error", "Failed to evaluate model. Check console for details.")
                return
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Update metric labels
            self.accuracy_label.configure(text=f"Accuracy: {accuracy:.2%}")
            self.precision_label.configure(text=f"Precision: {precision:.2%}")
            self.recall_label.configure(text=f"Recall: {recall:.2%}")
            self.f1_label.configure(text=f"F1 Score: {f1:.2%}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            self.canvas.draw()
            
            # Print detailed classification report
            report = classification_report(y_test, y_pred)
            print(f"\nDetailed Classification Report for {model_name}:")
            print(report)
            
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
            print(f"Error during evaluation: {e}")
