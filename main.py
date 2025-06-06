import tkinter as tk
from datetime import date, datetime, timedelta
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings

# Importing functions from prediction module
from prediction import predict, get_today, get_history_data

# Get today's weather data
tmax, tmin, temp, fmax, fmin, feel, hum, precip, wind, cloud, vis, srise, sset, moon, gust, uv, dew = get_today()

# Retrieve historical data and predictions
hist = get_history_data()
print(hist.columns)
prediction = predict()

# Append predictions to historical data
hist = hist._append(prediction, ignore_index=True)
print(hist)

td = date.today()  # Today's date

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Weather Forecast")
root.resizable(False, False)

# Create a canvas to draw UI elements
CANVAS = tk.Canvas(root, width=1000, height=600)
CANVAS.pack()

def print_size():
    print(f"Width: {root.winfo_width()}, Height: {root.winfo_height()}")

# root.after(1000, print_size)


# Function to interpret cloud cover percentage
def cloud_convert(cloud):
    if 89 < cloud:
        return "Cloudy"
    elif cloud < 69:
        return "Mostly Cloudy"
    elif cloud < 29:
        return "Partly Sunny"
    elif cloud < 9:
        return "Mostly Sunny"
    else:
        return "Clear"

# Function to draw text on the canvas
def draw_text(coord, text, size, start="center", fill="black"):
    return CANVAS.create_text(coord[0], coord[1], text=text, font=("Arial", size), anchor=start, fill=fill)

# Function to convert date/time strings to different formats
def convert_time(date, format="date"):
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    if format == "date":
        return date_obj.strftime("%a, %b %d, %Y")
    elif format == "day":
        return date_obj.strftime("%d %b, %Y")
    elif format == "no_year":
        return date_obj.strftime("%a %d %b")
    elif format == "min":
        return date_obj.strftime("%H:%M")
    elif format == "fraction":
        return (int(date_obj.strftime("%H")) + int(date_obj.strftime("%M")) / 60) / 24

# Function to draw a rectangular box
def draw_box(x1, x2, y1, y2, fill_color="white", border_color="black", border_width=2):
    CANVAS.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=border_color, width=border_width)

# Function to create a color spectrum for temperature
def create_color_spectrum(CANVAS, width, height, x, y, fraction):
    num_lines = 300  # Number of lines in the spectrum
    # Define color gradient stops from cool to warm
    colors = [
        (0, 0, 255),    # Blue
        (0, 255, 255),  # Cyan
        (0, 255, 0),    # Green
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
        (255, 0, 0)     # Red
    ]

    # Function to interpolate between colors
    def interpolate_color(color1, color2, t):
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        return f'#{r:02x}{g:02x}{b:02x}'

    num_colors = len(colors) - 1
    lines_per_color = num_lines // num_colors

    # Draw gradient lines
    for i in range(num_colors):
        for j in range(lines_per_color):
            t = j / lines_per_color
            color = interpolate_color(colors[i], colors[i + 1], t)
            x0 = x + ((i * lines_per_color) + j) / num_lines * width
            CANVAS.create_line(x0, y, x0, y + height, fill=color)

    # Draw a border around the spectrum
    CANVAS.create_rectangle(x, y, x + width, y + height, outline="black", width=2)

    # Draw stationary pointer lines for each fraction
    for i in fraction:
        pointer_position = x + i * width
        CANVAS.create_line(pointer_position, y, pointer_position, y + height, fill="black", width=2)

# Function to update text in the canvas
def update_text(item_id, new_text):
    CANVAS.itemconfigure(item_id, text=new_text)

# Function to create a horizontal line with markers
def create_line(x1, y1, x2, y2, fraction1, fraction2, small_line_length=6, offset1=-3, offset2=3, \
line_color='white', small_line_color='red'):
    CANVAS.create_line(x1, y1, x2, y2, fill=line_color, width=3)

    # Add small lines for fractions
    small_line1_x = x1 + fraction1 * (x2 - x1)
    small_line2_x = x1 + fraction2 * (x2 - x1)
    CANVAS.create_line(small_line1_x, y1 + offset1 - small_line_length, small_line1_x, 
    y1 + offset1 + small_line_length, fill=small_line_color, width=2)
    CANVAS.create_line(small_line2_x, y1 + offset2 - small_line_length, small_line2_x, 
    y1 + offset2 + small_line_length, fill=small_line_color, width=2)

# Function to create a line graph
def create_line_graph(data, width, height, x, y, ylabel=None):
    fig = Figure(figsize=(width, height), dpi=100)
    plot = fig.add_subplot(1, 1, 1)
    x_data, y_data = data
    plot.plot(x_data, y_data)
    plot.set_ylabel(ylabel)
    plot.tick_params(axis='both', which='major', labelsize=7.5)

    # Create a frame to embed the graph in Tkinter
    frame = ttk.Frame(root)
    frame.place(x=x, y=y, width=width * 100, height=height * 100)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    return canvas, plot

# Function to update the line graph with new data
def update_graph(data, canvas, plot, ylabel):
    x_data, y_data = data
    plot.clear()
    plot.plot(x_data, y_data)
    plot.set_ylabel(ylabel)
    canvas.draw()


def main():

    draw_box(10, 400, 65, 590, fill_color="lightblue") #TODAY BOX
    draw_box(410, 990, 65, 230, fill_color="lightblue") #SUMMARY BOX
    draw_box(410, 600, 240, 590, fill_color="lightblue") #TMR BOX
    draw_box(610, 800, 240, 590, fill_color="lightblue") #2 DAYS AFTER BOX
    draw_box(810, 990, 240, 590, fill_color="lightblue") #3 DAYS AFTER BOX


    draw_box(20, 250, 225, 360, fill_color="#414150") #darkblue: 414150
    draw_box(260, 390, 225, 360, fill_color="#414150")

    draw_box(20, 140, 370, 470, fill_color="#414150")  
    draw_box(150, 280, 370, 470, fill_color="#414150") 
    draw_box(290, 390, 370, 470, fill_color="#414150") 

    draw_box(20, 140, 480, 580, fill_color="#414150")  
    draw_box(150, 280, 480, 580, fill_color="#414150") 
    draw_box(290, 390, 480, 580, fill_color="#414150") 





    draw_text((195, 25), f"Coventry", 36)
    draw_text((195, 100), f"{convert_time(str(td), 'no_year')}", 20)
    draw_text((195+10, 140), f"{round(temp, 1)}°", 32)
    draw_text((195, 172), f"{cloud_convert(cloud)}", 12)
    draw_text((150, 195), f"L:{round(tmin, 1)}°C", 12)
    draw_text((225, 195), f"H:{round(tmax, 1)}°C", 12)


    # FIRST BOX: TEMPERATURE + FEELS LIKE
    def temp0(event): 
        selected_option = combobox_temp0.get()
        if selected_option == "Temperature":
            t, ti, ta = temp, tmin, tmax
        else:
            t, ti, ta = feel, fmin, fmax


        # print(temp, feel)
        update_text(temp_id, f"Average Temperature: {t}")
        update_text(fluctuate_id, f"Fluctuate: {ti}~{ta}")

        return t, ti, ta


    temp_id = draw_text((30, 275), f"Average Temperature: {temp}", 12, start="w", fill="white")
    fluctuate_id = draw_text((30, 295), f"Fluctuate: {tmin}~{tmax}", 12, start="w", fill="white")


    options_temp0 = ["Temperature", "Feels like"]

    combobox_temp0 = ttk.Combobox(root, values=options_temp0, state='readonly')
    combobox_temp0.set("Temperature")  # Set the default value

    combobox_temp0.bind("<<ComboboxSelected>>", temp0)
    combobox_temp0.place(x=30, y=240, width=150)


    create_color_spectrum(CANVAS, 200, 20, 30, 320, [(tmin+10)/50, (tmax+10)/50, (temp+10)/50])


    #SECOND BOX: SUN RISE + SUN SET
    create_line(270, 290, 360, 290, convert_time(srise, "fraction"), convert_time(sset, "fraction"))

    draw_text((270, 250), "Sun rises at:", 12, start="w", fill="white")
    draw_text((270, 330), "and sets at:", 12, start="w", fill="white")


    draw_text((270, 270), f"{convert_time(srise, 'min')}", 12, start="w", fill="white")
    draw_text((320, 310), f"{convert_time(sset, 'min')}", 12, start="w", fill="white")


    #THIRD BOX: UV INDEX

    draw_text((30, 400), f"UV Index: {uv}", 12, start="w", fill="white")

    create_color_spectrum(CANVAS, 100, 20, 25, 430, [uv/12])

    # FOURTH BOX: WIND + GUST
    draw_text((153, 400), f"Wind speed up to:\n{wind} km/h", 12, start="w", fill="white")
    draw_text((153, 440), f"Wind gust up to: \n{gust} km/h", 12, start="w", fill="white")


    # SIXTH BOX: HUMIDITY + DEW
    draw_text((30, 490), f"(Average)", 12, start="w", fill="white")
    draw_text((30, 520), f" - Humidity: \n{hum}%", 12, start="w", fill="white")
    draw_text((30, 560), f" - Dew point: \n{dew} °C", 12, start="w", fill="white")



    # FIFTH BOX: visibility
    draw_text((300, 420), f"Average \n visibility \n  is about: \n   {vis}km", 12, start="w", fill="white")

    # SEVENTH BOX: precip + cloud cover
    draw_text((153, 510), f"Total precipitation: \n{precip} mm", 12, start="w", fill="white")
    draw_text((153, 550), f"Total cloud cover: \n{cloud}%", 12, start="w", fill="white")


    # EIGHTH BOX: moon phase
    draw_text((300, 530), f"Moon phase: \n{moon}", 12, start="w", fill="white")


    def update_data(): 
        selected_option = combobox_pred.get()

        convert_dict =  {"Temperature" :"temp", 
                         "Max Temperature" : "tempmax", 
                         "Min Temperature" : "tempmin", 
                         "Feels like" : "feelslike", 
                         "Max Feels like" : "feelslikemax", 
                         "Min Feels like" : "feelslikemin", 
                         "Humidity" : "humidity", 
                         "Precipitation" : "precip", 
                         "Wind Speed" : "windspeed", 
                         "Wind gust speed" : "windgust", 
                         "Moon phase" : "moonphase", 
                         "UV Index" : "uvindex", 
                         "Visibility" : "visibility"}
        pred_data = [range(1, 11, 1) ,hist.loc[:, convert_dict[selected_option]].to_list()]

        update_graph(pred_data, sub_canvas, sub_plot, ylabel=selected_option)
        

    options_pred = ["Temperature", "Max Temperature", "Min Temperature", "Feels like", "Max Feels like",
     "Min Feels like", "Humidity", "Precipitation", "Wind Speed", "Wind gust speed", "Moon phase", "UV Index", "Visibility"]

    combobox_pred = ttk.Combobox(root, values=options_pred, state='readonly')
    combobox_pred.set("Temperature")  # Set the default value

    combobox_pred.bind("<<ComboboxSelected>>", lambda event: update_data())
    combobox_pred.place(x=837, y=20, width=150)

    pred_data = [range(1, 11, 1) ,hist.loc[:, "temp"].to_list()]
    sub_canvas, sub_plot = create_line_graph(pred_data, 5.8, 1.65, 410, 65, ylabel="Temperature")

    draw_text((500, 270), f"{convert_time(str(td+timedelta(days=1)), 'no_year')}", 19)
    draw_text((700, 270), f"{convert_time(str(td+timedelta(days=2)), 'no_year')}", 19)
    draw_text((900, 270), f"{convert_time(str(td+timedelta(days=3)), 'no_year')}", 19)

    for i in range(3):

        draw_text((500+i*200, 300), f"Temperature: {round(hist.loc[7+i, 'temp'], 2)}°C", 12)
        draw_text((500+i*200, 320), f"Fluctuate: {round(hist.loc[7+i, 'tempmin'], 2)}~{round(hist.loc[7+i, 'tempmax'], 2)}°C", 12)
        draw_text((500+i*200, 340), f"Feels like: {round(hist.loc[7+i, 'feelslike'], 2)}°C", 12)
        draw_text((500+i*200, 360), 
        f"Fluctuate: {round(hist.loc[7+i, 'feelslikemin'], 2)}~{round(hist.loc[7+i, 'feelslikemax'], 2)}°C", 12)

        sunrise, sunset = hist.loc[7+i, 'sunrise']*24, hist.loc[7+i, 'sunset']*24
        draw_text((500+i*200, 380), f"Sun rises at: {round(sunrise)}:{round(abs(sunrise - round(sunrise))*60)}", 12)
        draw_text((500+i*200, 400), f"Sun sets at: {round(sunset)}:{round(abs(sunset - round(sunset))*60)}", 12)

        draw_text((500+i*200, 420), f"Humidity:  {round(hist.loc[7+i, 'humidity'], 2)}%", 12)
        draw_text((500+i*200, 440), f"Wind speed:  {round(hist.loc[7+i, 'windspeed'], 2)} km/h", 12)
        draw_text((500+i*200, 460), f"Gust speed:  {round(hist.loc[7+i, 'windgust'], 2)}km/h", 12)

        draw_text((500+i*200, 480), f"Precipitation:  {round(hist.loc[7+i, 'precip'], 2)}mm", 12)
        draw_text((500+i*200, 500), f"Cloud cover:  {round(hist.loc[7+i, 'cloudcover'], 2)}%", 12)
        draw_text((500+i*200, 520), f"UV Index:  {round(hist.loc[7+i, 'uvindex'], 2)}", 12)

        draw_text((500+i*200, 540), f"Visibility:  {round(hist.loc[7+i, 'visibility'], 2)}km", 12)
        draw_text((520+i*200, 560), f"Moon phase:  {round(hist.loc[7+i, 'moonphase'], 2)}", 12)

    
    root.mainloop()


if __name__ == "__main__":
    main()