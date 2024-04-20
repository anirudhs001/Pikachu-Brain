import socket
import os
from PIL import Image
import io
import re
from glob import glob
from tkinter import Tk, PhotoImage, Label
import math
import time
import cv2
import queue
import threading
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def resize_image(image):
    max_hw, min_hw = max(image.size), min(image.size)
    aspect_ratio = max_hw / min_hw
    max_len, min_len = 800, 400
    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = image.size
    if H > W:
        H, W = longest_edge, shortest_edge
    else:
        H, W = shortest_edge, longest_edge
    image = image.resize((W, H))
    return image


def send_text_and_image(
    client_socket,
    text_message,
    image_path,
):
    # Send text message
    client_socket.sendall(text_message.encode("utf-8"))

    # Send image
    if image_path:
        image = Image.open(image_path)
        print("Original image size: ", image.size)
        image = resize_image(image)
        print("Resized image size: ", image.size)
        # convert PIL Image to byte stream
        output_buffer = io.BytesIO()  # in-memory buffer
        image.save(output_buffer, format="JPEG")
        image_data = output_buffer.getvalue()  # byte stream
        image_size = str(len(image_data)).ljust(20)
        print("Image bytes", image_size)
        client_socket.sendall(image_size.encode("utf-8"))
        client_socket.sendall(image_data)
    else:
        client_socket.sendall("0".encode("utf-8"))

    # client_socket.close()


# Function to continuously update and display the image
def update_image(imgplot, img):
    imgplot.set_data(img)
    plt.draw()
    plt.pause(0.001)  # Pause to update the plot


if __name__ == "__main__":
    # Example usage

    server_address = "localhost"
    server_port = 5555
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, server_port))
    image_queue = queue.Queue()

    # Create a thread to continuously update and display the image
    # image_thread = threading.Thread(target=update_image, args=(image_queue,))
    # image_thread.daemon = True  # Set the thread as daemon so it will be terminated when the main thread exits

    # Start the image thread
    # image_thread.start()

    # load images in mem
    num_images = len(glob("faces/*.png"))
    images = [mpimg.imread(f"faces/Picture {i}.png") for i in range(1, num_images)]
    imgplot = plt.imshow(images[0])
    plt.ion()  # Turn interactive mode on

    # panel = Label(window, image=images[0])
    # panel.pack()
    # window.mainloop()

    for i in range(10):
        message = input("msg> ")
        if i == 0:
            image_path = (
                "/Users/anirudhsingh/MISC/playground/Pikachu/Pikachu-Brain/image.png"
            )
        else:
            image_path = None

        send_text_and_image(client_socket, message, image_path)
        # receive response from server
        resp = client_socket.recv(1024).decode("utf-8")
        pattern = r"<face:(\d+)>"
        words = resp.split(" ")
        print("res> ", end="")
        for word in words:
            if "<face" in word:
                face_idx = int(re.findall(pattern, resp)[0])
                # image_queue.put(images[face_idx])
                update_image(imgplot, images[face_idx])
                # cv2.imshow("image", images[face_idx])
                # cv2.waitKey(0)
            else:
                print(word, end=" ")
                time.sleep(0.1)
        print()

    print("FIN.")
    client_socket.close()
