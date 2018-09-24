from tkinter import *
import tensorflow as tf
from PIL import Image, ImageGrab, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = PhotoImage(master=canvas, width=figure_w, height=figure_h)

    canvas.create_image(loc[0] + figure_w / 2, loc[1] + figure_h / 2, image=photo)
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    return photo


class GUI(object):
    def __init__(self):

        self.root = Tk()
        self.root.winfo_toplevel().title("Digit classifier")
        self.root.geometry('{}x{}'.format(650, 700))
        self.root.resizable(width=False, height=False)

        # GUI widgets
        self.choose_size_button = Scale(self.root, from_=12, to=20, width=12, orient=HORIZONTAL, length=80)
        self.choose_size_button.grid(row=0, column=0, sticky=E)

        self.clear_button = Button(self.root, text='Clear', font=("", 12), command=self.use_clear)
        self.clear_button.grid(row=0, column=2, sticky=W)

        self.canvas = Canvas(self.root, bg='white', width=168, height=168)
        self.canvas.grid(row=1, column=1, columnspan=1)

        self.Text = Label(self.root, text="", font=("", 20))
        self.Text.grid(row=0, columnspan=3)

        self.old_x = None
        self.old_y = None

        self.canvas2 = Canvas(self.root, width=700, height=600)
        self.canvas2.grid(row=2, columnspan=3)

        # Setup plots
        dummy_image = Image.new('L', (28, 28))

        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
        ax1.set_title("Inverted")
        ax1.imshow(dummy_image)
        ax1.axis('off')

        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=1)
        ax2.set_title("Cropped & downscaled")
        ax2.imshow(dummy_image)
        ax2.axis('off')

        ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1)
        ax3.set_title("Centered")
        ax3.imshow(dummy_image)
        ax3.axis('off')

        ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        ax4.set_title("Probabilities", fontsize=18)
        ax4.bar(np.arange(10), 0, 0.2)

        self.figure = plt.gcf()
        self.fig_photo = draw_figure(self.canvas2, self.figure)
        plt.close()

        self.images = [ax1, ax2, ax3]
        self.bar = ax4

        # Load model
        print("Restoring model...")
        self.model = tf.keras.models.load_model('model.h5')
        print("Model restored...")

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.guess_digit)

    def use_clear(self):
        self.canvas.delete("all")
        self.change_text("")

    def change_text(self, text=""):
        self.Text['text'] = text

    def paint(self, event):
        line_width = self.choose_size_button.get()
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=line_width, fill='black',
                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def guess_digit(self, event):
        self.old_x, self.old_y = None, None

        # Extract image from canvas
        x = self.canvas.winfo_rootx() + 2
        y = self.canvas.winfo_rooty() + 2
        x1 = x + self.canvas.winfo_width() - 4
        y1 = y + self.canvas.winfo_height() - 4
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
        img = ImageOps.invert(img)
        self.images[0].imshow(img)  # Plot inverted image

        # Crop image
        arr = np.array(img)
        x, y = 0, 0
        max_x, max_y = arr.shape[0] - 1, arr.shape[1] - 1

        while np.sum(arr[:y + 2]) == 0:
            y += 1
        while np.sum(arr[max_y - 2:]) == 0:
            max_y -= 1
        while np.sum(arr[:, :x + 2]) == 0:
            x += 1
        while np.sum(arr[:, max_x - 2:]) == 0:
            max_x -= 1

        cropped = img.crop((x, y, max_x, max_y))

        # Down-scale image while keeping aspect ratio
        downscaled = cropped
        rows, cols = downscaled.height, downscaled.width
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            downscaled = downscaled.resize((cols, rows), Image.ANTIALIAS)
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            downscaled = downscaled.resize((cols, rows), Image.ANTIALIAS)

        # Paste the digit onto a black 28x28 image
        centered = Image.new("L", (28, 28))
        centered.paste(downscaled, (4, 4))

        self.images[1].imshow(centered)

        # Scale pixel values to a range of 0 to 1
        centered = np.array(centered) / 255.0

        # Center digit in image using Center of Mass
        c1 = ndimage.center_of_mass(np.ones_like(centered))
        cm = ndimage.center_of_mass(centered)
        centered = np.roll(centered, int(round((c1[0] - cm[0]))), axis=0)
        centered = np.roll(centered, int(round((c1[1] - cm[1]))), axis=1)
        self.images[2].imshow(centered, cmap='gray')

        # Get the model's prediction
        inp = np.reshape(centered, (1, 28, 28))
        prediction = self.model.predict_classes(inp)
        p = self.model.predict(inp)
        self.change_text(prediction[0])

        # Plot image and prediction rates
        self.bar.cla()
        p = np.power(p, (1 / 10))
        plt.xticks(np.arange(len(p[0])))
        self.bar.bar(np.arange(len(p[0])), p[0], 0.2)
        self.fig_photo = draw_figure(self.canvas2, self.figure)


if __name__ == "__main__":
    GUI()