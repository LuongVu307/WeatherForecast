import tkinter as tk
from datetime import date, datetime, timedelta
from prediction import predict, get_today

import warnings
warnings.filterwarnings("ignore")

td = date.today()

root = tk.Tk()
root.title("Weather Forcast")
root.resizable(False, False)

background_image = tk.PhotoImage(file="background.png")


canvas = tk.Canvas(root, width=1000, height=600)
canvas.pack()


# canvas.create_image(0, 0, anchor="nw", image=background_image)

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
    
def draw_text(coord, text, size, start="center", fill="black"):
    canvas.create_text(coord[0], coord[1], text=text, font=("Arial", size), anchor=start, fill=fill)

def convert_time(date, format="date"):
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    if format=="date":
        return date_obj.strftime("%a, %b %d, %Y")
    elif format=="day":
        return date_obj.strftime("%d %b, %Y")
    elif format=="no_year":
        return date_obj.strftime("%a %d %b")
    elif format=="min":
        return date_obj.strftime("%H:%M")

    
def draw_box(x1, x2, y1, y2, fill_color="white", border_color="black", border_width=2):
    canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=border_color, width=border_width)

tmin, tmax, temp, hum, precip, wind, cloud, uv, fmax, fmin, feel, dew, gust, pres, vis, srise, sset, wdir = get_today()

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

# draw_text((20, 250), f"Humidity: {hum}%", 15, start="w",fill="white")
# draw_text((20, 275), f"Precipitation: {precip}mm", 15, start="w", fill="white")
# draw_text((20, 300), f"Windspeed: {round(wind-2.5, 1)}-{round(wind+2.5, 1)}km/h", 15, start="w", fill="white")
# draw_text((20, 325), f"Cloud Cover: {cloud}%", 15, start="w", fill="white")
# draw_text((20, 350), f"UV Index: {uv}", 15, start="w", fill="white")

draw_text((30, 280), f"{wind}", 18, start="w", fill="white")
draw_text((30, 320), f"{gust}", 18, start="w", fill="white")

draw_text((270, 280), f"{convert_time(srise, 'min')} -", 14, start="w", fill="white")
draw_text((330, 280), f"{convert_time(sset, 'min')}", 14, start="w", fill="white")

draw_text((30, 400), f"{uv}", 16, start="w", fill="white")
draw_text((160, 400), f"{fmin} - {fmax}", 14, start="w", fill="white")
draw_text((160, 425), f"Avg: {feel}", 14, start="w", fill="white")
draw_text((300, 400), f"{precip}", 16, start="w", fill="white")

draw_text((30, 500), f"{hum}", 16, start="w", fill="white")
draw_text((30, 520), f"{dew}", 13, start="w", fill="white")
draw_text((160, 500), f"{vis}", 16, start="w", fill="white")
draw_text((300, 500), f"{pres}", 16, start="w", fill="white")






root.mainloop()
