    from tkinter import *
    import tensorflow as tf
    from PIL import Image, ImageGrab
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage

    class GUI(object):
        def __init__(self):
            self.root = Tk()
            self.root.winfo_toplevel().title("Digit classifier")
            self.root.geometry('{}x{}'.format(300, 250))
            self.root.resizable(width=False, height=False)

            # GUI widgets
            self.choose_size_button = Scale(self.root, from_=12, to=20, width=12, orient=HORIZONTAL, length=80)
            self.choose_size_button.grid(row=0, column=0, columnspan=2, padx=60)

            self.clear_button = Button(self.root, text='Clear', command=self.use_clear)
            self.clear_button.grid(row=0, column=2)

            self.c = Canvas(self.root, bg='black', width=168, height=168)
            self.c.grid(row=1, columnspan=3, padx=60)

            self.Text = Label(self.root, text="", font=("", 20))
            self.Text.grid(row=2, columnspan=3)

            self.old_x = None
            self.old_y = None

            # Load model
            print("Restoring model...")
            self.model = tf.keras.models.load_model('model.h5')
            print("Model restored...")

            self.setup()
            self.root.mainloop()

        def setup(self):
            self.c.bind('<B1-Motion>', self.paint)
            self.c.bind('<ButtonRelease-1>', self.guess_digit)

        def use_clear(self):
            self.c.delete("all")
            self.change_text("")

        def change_text(self, text=""):
            self.Text['text'] = text

        def paint(self, event):
            line_width = self.choose_size_button.get()
            if self.old_x and self.old_y:
                self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=line_width,fill='white',
                                   capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.old_x = event.x
            self.old_y = event.y

        # Extract and pre-process digit
        def process_image(self):
            # Extract image from canvas
            x = self.c.winfo_rootx()+2
            y = self.c.winfo_rooty()+2
            x1 = x + self.c.winfo_width()-4
            y1 = y + self.c.winfo_height()-4
            img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

            plt.figure(6, (8, 6))
            plt.arrow(20, 2, 0.5, 0.5)
            plt.suptitle("Image pre-processing", fontsize=18)
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            plt.cla()
            plt.subplot(2, 4, 1)
            plt.title("Original")
            plt.imshow(img)

            # Keep deleting black sides to remove excess black pixels around the
            arr = np.array(img)
            x, y = 0, 0
            maxX, maxY = arr.shape[0]-1, arr.shape[1]-1

            while np.sum(arr[:y+2]) == 0:
                y+=1
            while np.sum(arr[maxY-2:]) == 0:
                maxY-=1
            while np.sum(arr[:, :x+2]) == 0:
                x+=1
            while np.sum(arr[:, maxX-2:]) == 0:
                maxX -= 1

            extracted = img.crop((x, y, maxX, maxY))
            plt.subplot(2, 4, 2)
            plt.title("Cropped")
            plt.imshow(extracted)

            # Down-scale image while keeping aspect ratio
            rows, cols = extracted.height, extracted.width
            if rows > cols:
                factor = 20.0 / rows
                rows = 20
                cols = int(round(cols * factor))
                extracted = extracted.resize((cols, rows), Image.ANTIALIAS)
            else:
                factor = 20.0 / cols
                cols = 20
                rows = int(round(rows * factor))
                extracted = extracted.resize((cols, rows), Image.ANTIALIAS)

            # Paste the digit onto a black 28x28 image
            processed = Image.new("L", (28, 28))
            processed.paste(extracted, (4, 4))

            plt.subplot(2, 4, 3)
            plt.title("Downscaled")
            plt.imshow(processed)


            # Downscale pixel values to a range of 0 to 1
            processed = np.array(processed) / 255.0

            # Center digit in image using Center of Mass
            c1 = ndimage.center_of_mass(np.ones_like(processed))
            cm = ndimage.center_of_mass(processed)
            processed = np.roll(processed, int(round((c1[0] - cm[0]))), axis=0)
            processed = np.roll(processed, int(round((c1[1] - cm[1]))), axis=1)

            plt.subplot(2, 4, 4)
            plt.title("Centered")
            plt.imshow(processed, cmap='gray')

            return processed

        def guess_digit(self, event):
            self.old_x, self.old_y = None, None

            # Extract digit
            img = self.process_image()
            inp = np.reshape(img, (1, 28, 28))

            # Get the model's prediction
            prediction = self.model.predict_classes(inp)
            p = self.model.predict(inp)

            print(p)
            p = np.power(p, (1/10))

            # Plot image and prediction rates
            self.change_text(prediction[0])
            plt.subplot(2, 1, 2)
            plt.title("Probabilities", fontsize=18)
            plt.xticks(np.arange(len(p[0])))
            #plt.ylim((0,0.3))
            plt.bar(np.arange(len(p[0])), p[0], 0.2)
            plt.draw()


    if __name__ == "__main__":
        GUI()