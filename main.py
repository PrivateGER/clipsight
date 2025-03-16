#!/usr/bin/env python3
"""
CLIP Image Search

A graphical user interface for searching images using CLIP embeddings.
"""

import tkinter as tk
from app import CLIPSearchApp

if __name__ == "__main__":
    root = tk.Tk()
    app = CLIPSearchApp(root)
    root.mainloop() 