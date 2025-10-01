
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, X):
        X = self.pool(F.relu(self.bn1(self.conv1(X))))
        X = F.relu(self.bn2(self.conv2(X)))
        X = self.pool(F.relu(self.bn3(self.conv3(X))))
        X = X.view(-1, 128 * 7 * 7)
        X = F.relu(self.bn_fc1(self.fc1(X)))
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


def load_checkpoint(model, filename="CheckPoint.pth", device="cpu"):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}) acc={checkpoint['best_acc']:.2f}%")
    return model


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# GUI
def run_interface():
    model = SimpleCNN()
    model = load_checkpoint(model, "CheckPoint.pth", device=device)

    root = tk.Tk()
    root.title("MNIST Digit Recognition")

    canvas = tk.Canvas(root, width=280, height=280, bg="white")
    canvas.grid(row=0, column=0, columnspan=4)

    label_result = tk.Label(root, text="Draw a digit and click Predict", font=("Arial", 16))
    label_result.grid(row=1, column=0, columnspan=4)


    def paint(event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)

    canvas.bind("<B1-Motion>", paint)


    def clear_canvas():
        canvas.delete("all")

    def predict():
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
    

        img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
    

        img = np.array(img)
    

        img = 255 - img
    

        img[img < 50] = 0
        img[img >= 50] = 255
    

        coords = np.column_stack(np.where(img > 0))
        if coords.size == 0:
            label_result.config(text="No digit found!", fg="red")
            return
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img = img[y_min:y_max+1, x_min:x_max+1]
    

        pil_img = Image.fromarray(img).resize((20, 20), Image.LANCZOS)
    

        new_img = Image.new("L", (28, 28), 0)
        new_img.paste(pil_img, ((28 - 20) // 2, (28 - 20) // 2))
    
        new_img.save("debug_processed.png")  # 🔍
    

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(new_img).unsqueeze(0).to(device)
    
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            print(f"Predicted Digit: {predicted.item()}")
    
        label_result.config(text=f"Predicted Digit: {predicted.item()}", fg="blue")
    
    btn_predict = tk.Button(root, text="Predict", command=predict, font=("Arial", 14))
    btn_predict.grid(row=2, column=0, pady=10)

    btn_clear = tk.Button(root, text="Clear", command=clear_canvas, font=("Arial", 14))
    btn_clear.grid(row=2, column=1, pady=10)

    btn_quit = tk.Button(root, text="Quit", command=root.destroy, font=("Arial", 14))
    btn_quit.grid(row=2, column=2, pady=10)

    root.mainloop()


if __name__ == "__main__":
    run_interface()