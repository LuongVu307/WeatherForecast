import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random

# Function to create a line graph
def create_line_graph(data, width, height, ylabel=None):
    fig = Figure(figsize=(width, height), dpi=100)
    plot = fig.add_subplot(1, 1, 1)

    # Unpack data into x and y components
    x_data, y_data = data

    # Plot the data
    plot.plot(x_data, y_data)

    if ylabel:
        plot.set_ylabel(ylabel)
        
    plot.tick_params(axis='both', which='major', labelsize=7.5)

    return fig, plot

# Function to update the graph with new data
def update_graph(data, canvas, plot):
    x_data, y_data = data

    # Clear the previous plot
    plot.clear()
    
    # Plot the new data
    plot.plot(x_data, y_data)
    
    # Redraw the canvas
    canvas.draw()

# Sample usage
def main():
    root = tk.Tk()
    root.title("Dynamic Line Graph")

    # Initial data
    x_data = list(range(10))
    y_data = [random.randint(0, 10) for _ in range(10)]

    # Create the initial graph
    fig, plot = create_line_graph((x_data, y_data), width=5, height=4, ylabel='Value')
    
    # Create a frame for the plot
    frame = ttk.Frame(root)
    frame.place(x=10, y=50, width=500, height=400)  # Adjust size with 100 as a scaling factor

    # Create a canvas to display the figure
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create a ComboBox for selecting options
    options = ['Option 1', 'Option 2', 'Option 3']
    selected_option = tk.StringVar()
    combobox = ttk.Combobox(root, textvariable=selected_option, values=options)
    combobox.place(x=10, y=10)
    combobox.current(0)

    # Function to update data based on selected option
    def update_data():
        selected = selected_option.get()
        if selected == 'Option 1':
            y_data = [random.randint(0, 10) for _ in range(10)]
        elif selected == 'Option 2':
            y_data = [random.randint(10, 20) for _ in range(10)]
        elif selected == 'Option 3':
            y_data = [random.randint(20, 30) for _ in range(10)]
        
        # Update the graph
        update_graph((x_data, y_data), canvas, plot)

    # Bind the ComboBox selection event to update the graph
    combobox.bind('<<ComboboxSelected>>', lambda event: update_data())

    root.mainloop()

if __name__ == "__main__":
    main()
