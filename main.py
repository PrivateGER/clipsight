#!/usr/bin/env python3
"""
CLIP Image Search

A graphical user interface for searching images using CLIP embeddings.
"""

import tkinter as tk
import sys
import darkdetect
from app import CLIPSearchApp
import sv_ttk

# Only import and define Windows-specific functionality on Windows
if sys.platform == 'win32':
    import pywinstyles
    
    def apply_theme_to_titlebar(root):
        version = sys.getwindowsversion()

        if version.major == 10 and version.build >= 22000:
            # Set the title bar color to the background color on Windows 11 for better appearance
            pywinstyles.change_header_color(root, "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa")
        elif version.major == 10:
            pywinstyles.apply_style(root, "dark" if sv_ttk.get_theme() == "dark" else "normal")

            # A hacky way to update the title bar's color on Windows 10 (it doesn't update instantly like on Windows 11)
            root.wm_attributes("-alpha", 0.99)
            root.wm_attributes("-alpha", 1)

if __name__ == "__main__":
    root = tk.Tk()
    app = CLIPSearchApp(root)
    
    # Apply theme
    sv_ttk.set_theme(darkdetect.theme())
    
    # Apply Windows-specific titlebar styling if on Windows
    if sys.platform == 'win32':
        apply_theme_to_titlebar(root)
    
    root.mainloop()
