import tkinter as tk
from tkinter import ttk
import sv_ttk

class ToolTip:
    """
    Create a tooltip for a given widget
    """
    def __init__(self, widget, text='', delay=500, wraplength=250):
        self.widget = widget
        self.text = text
        self.delay = delay  # Delay in milliseconds
        self.wraplength = wraplength  # Text wrapping length
        self.tooltip_window = None
        self.id = None
        self.clicked = False  # Track if tooltip was shown by click
        
        # Bind events
        self.widget.bind("<Enter>", self.schedule)
        self.widget.bind("<Leave>", self.hide)
        self.widget.bind("<Button-1>", self.show_on_click)  # Show immediately on click
        
    def schedule(self, event=None):
        """Schedule showing the tooltip"""
        self.hide()
        self.clicked = False
        self.id = self.widget.after(self.delay, self.show)
    
    def show_on_click(self, event=None):
        """Show tooltip immediately when clicked"""
        if self.tooltip_window:
            self.hide()
            return
            
        self.clicked = True
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        self.show()
    
    def show(self):
        """Display the tooltip"""
        # Get widget position
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        # Creates a toplevel window
        self.tooltip_window = tk.Toplevel(self.widget)
        # Make it stay on top
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        # Determine colors based on theme
        current_theme = sv_ttk.get_theme()
        if current_theme == "dark":
            bg_color = "#333333"
            fg_color = "#ffffff"
        else:
            bg_color = "#ffffe0"  # Light yellow for light theme
            fg_color = "#000000"
        
        # Add a label with the tooltip text
        label = ttk.Label(self.tooltip_window, text=self.text, 
                          justify=tk.LEFT, wraplength=self.wraplength,
                          background=bg_color, foreground=fg_color,
                          relief=tk.SOLID, borderwidth=1)
        label.pack(padx=5, pady=5)
        
        # If shown by click, add a click outside to dismiss behavior
        if self.clicked:
            self.tooltip_window.bind("<Button-1>", self.hide)
            # Close after longer delay when clicked
            self.id = self.widget.after(5000, self.hide)
    
    def hide(self, event=None):
        """Hide the tooltip"""
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def update_text(self, text):
        """Update the tooltip text"""
        self.text = text 